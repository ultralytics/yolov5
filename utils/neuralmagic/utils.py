import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy
import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, download_framework_model_by_recipe_type
from sparsezoo import Model

from models.yolo import Model as Yolov5Model
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, check_dataset, check_yaml, colorstr
from utils.neuralmagic.quantization import _Add, update_model_bottlenecks
from utils.torch_utils import ModelEMA

__all__ = [
    "ALMOST_ONE",
    "sparsezoo_download",
    "ToggleableModelEMA",
    "load_ema",
    "load_sparsified_model",
    "neuralmagic_onnx_export",
    "export_sample_inputs_outputs",
]

SAVE_ROOT = Path.cwd()
RANK = int(os.getenv("RANK", -1))
ALMOST_ONE = 1 - 1e-9  # for incrementing epoch to be applied to recipe

# In previous integrations of NM YOLOv5, we were pickling models as long as they are
# not quantized. We've now changed to never pickling a model touched by us. This
# namespace hacking is meant to address backwards compatibility with previously
# pickled, pruned models.
import models
from models import common
setattr(common, "_Add", _Add)  # Definition of the _Add module has moved

# If using yolov5 as a repo and not a package, allow loading of models pickled w package
if "yolov5" not in sys.modules:
    sys.modules["yolov5"] = ""
    sys.modules["yolov5.models"] = models
    sys.modules["yolov5.models.common"] = common


class ToggleableModelEMA(ModelEMA):
    """
    Subclasses YOLOv5 ModelEMA to enabled disabling during QAT
    """

    def __init__(self, enabled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = enabled

    def update(self, *args, **kwargs):
        if self.enabled:
            super().update(*args, **kwargs)


def sparsezoo_download(path: str, recipe: Optional[str] = None) -> str:
    """
    Loads model from the SparseZoo and override the path with the new download path
    """
    return download_framework_model_by_recipe_type(
        Model(path), recipe, "pt"
    )


def load_ema(
    ema_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    enabled: bool = True,
    **ema_kwargs,
) -> ToggleableModelEMA:
    """
    Loads a ToggleableModelEMA object from a ModelEMA state dict and loaded model
    """
    ema = ToggleableModelEMA(enabled, model, **ema_kwargs)
    ema.ema.load_state_dict(ema_state_dict)
    return ema


def load_sparsified_model(
    ckpt: Union[Dict[str, Any], str], device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module:
    """
    From a sparisifed checkpoint, loads a model with the saved weights and
    sparsification recipe applied

    :param ckpt: either a loaded checkpoint or the path to a saved checkpoint
    :param device: device to load the model onto
    """
    nm_log_console("Loading sparsified model")

    # Load checkpoint if not yet loaded
    ckpt = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location=device)

    if isinstance(ckpt["model"], torch.nn.Module):
        model = ckpt["model"]

    else:
        # Construct randomly initialized model model and apply sparse structure modifiers
        model = Yolov5Model(ckpt.get("yaml"))
        model = update_model_bottlenecks(model).to(device)
        checkpoint_manager = ScheduledModifierManager.from_yaml(
            ckpt["checkpoint_recipe"]
        )
        checkpoint_manager.apply_structure(
            model, ckpt["epoch"] + ALMOST_ONE if ckpt["epoch"] >= 0 else float("inf")
        )

        # Load state dict
        model.load_state_dict(ckpt["ema"] or ckpt["model"], strict=True)
        model.hyp = ckpt.get("hyp")
        model.nc = ckpt.get("nc")

    model.sparsified = True
    model.float()

    return model


def nm_log_console(message: str, logger: "Logger" = None, level: str = "info"):
    """
    Log sparsification-related messages to the console

    :param message: message to be logged
    :param level: level to be logged at
    """
    # default to global logger if none provided
    logger = logger or LOGGER

    if RANK in [0, -1]:
        if level == "warning":
            logger.warning(
                f"{colorstr('Neural Magic: ')}{colorstr('yellow', 'warning - ')}"
                f"{message}"
            )
        else:  # default to info
            logger.info(f"{colorstr('Neural Magic: ')}{message}")


def neuralmagic_onnx_export(
    model: torch.nn.Module,
    sample_data: torch.Tensor,
    weights_path: Path,
    one_shot: Optional[str],
    dynamic: Optional[Dict[str, int]],
    output_names: List[str],
) -> Path:
    """
    Augmented ONNX export to optimize and properly post-process sparsified models

    :param model: model to export
    :param sample_data: data to be used with export
    :weights_path: path from which the torch model was loaded. Used only for save
        pathing and naming purposes
    :one_shot: one_shot recipe, if one was applied. Used only for save pathing and
        naming purposes
    :param dynamic: dictionary of input or output names to list of dimensions
        of those tensors that should be exported as dynamic
    :output_names: names of output tensors
    :return: path to saved ONNX model
    """

    # If the target model is a SparseZoo or YOLOv5 Hub model, save it to a
    # DeepSparse_Deployment directory at the working directory root. Inside, create a
    # subdirectory based on model/stub name
    if str(weights_path).startswith("zoo:") or not len(weights_path.parents):
        sub_dir = (
            str(weights_path).split("zoo:")[1].replace("/", "_")
            if str(weights_path).startswith("zoo:")
            else weights_path
        )

        # If one-shot applying a recipe, update name to convey the starting stub/model
        # and the one_shot recipe applied
        if one_shot:
            one_shot_str = str(weights_path).split("zoo:")[1].replace("/", "_")
            sub_dir = f"{sub_dir}_one_shot_{one_shot_str}"

        save_dir = Path(SAVE_ROOT) / "DeepSparse_Deployment" / sub_dir
        onnx_file_name = "model.onnx"

    else:
        save_dir = (
            weights_path.parents[1] / "DeepSparse_Deployment"
            if weights_path.parent.stem == "weights"
            else weights_path.parent / "DeepSparse_Deployment"
        )
        onnx_file_name = weights_path.with_suffix(".onnx").name

    save_dir.mkdir(parents=True, exist_ok=True)

    nm_log_console("Exporting model to ONNX format")

    # Use the SparseML custom onnx export flow for sparsified models
    exporter = ModuleExporter(model, save_dir.absolute())
    exporter.export_onnx(
        sample_data,
        name=onnx_file_name,
        convert_qat=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    saved_model_path = save_dir / onnx_file_name

    nm_log_console(f"Exported ONNX model to {saved_model_path}")

    return saved_model_path


def export_sample_inputs_outputs(
    dataset: Union[str, Path],
    data_path: str,
    model: torch.nn.Module,
    save_dir: Path,
    number_export_samples=100,
    image_size: int = 640,
    onnx_path: Union[str, Path, None] = None,
):
    """
    Export sample model input and output for testing with the DeepSparse Engine

    :param dataset: path to dataset to take samples from
    :param model: model to be exported. Used to generate outputs
    :param save_dir: directory to save samples to
    :param number_export_samples: number of samples to export
    :param image_size: image size
    :param onnx_path: Path to saved onnx model. Used to check if it uses uints8 inputs
    """

    nm_log_console(
        f"Exporting {number_export_samples} sample model inputs and outputs for "
        "testing with the DeepSparse Engine"
    )

    # Create dataloader
    data_dict = check_dataset(dataset, data_path)
    dataloader, _ = create_dataloader(
        path=data_dict["train"],
        imgsz=image_size,
        batch_size=1,
        stride=max(int(model.stride.max()), 32),
        hyp=model.hyp,
        augment=True,
        prefix=colorstr("train: "),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exported_samples = 0

    # Sample export directories
    sample_in_dir = save_dir / "sample_inputs"
    sample_out_dir = save_dir / "sample_outputs"
    sample_in_dir.mkdir(exist_ok=True)
    sample_out_dir.mkdir(exist_ok=True)

    save_inputs_as_uint8 = _graph_has_uint8_inputs(onnx_path) if onnx_path else False

    for images, _, _, _ in dataloader:
        # uint8 to float32, 0-255 to 0.0-1.0
        images = (images.float() / 255).to(device, non_blocking=True)
        model_out = model(images)

        if isinstance(model_out, tuple) and len(model_out) > 1:
            # Flatten into a single list
            model_out = [model_out[0], *model_out[1]]

        # Move to cpu for exporting
        images = images.detach().to("cpu")
        model_out = [elem.detach().to("cpu") for elem in model_out]

        outs_gen = zip(*model_out)

        for sample_in, sample_out in zip(images, outs_gen):

            sample_out = list(sample_out)

            file_idx = f"{exported_samples}".zfill(4)

            # Save inputs as numpy array
            sample_input_filename = sample_in_dir / f"inp-{file_idx}.npz"
            if save_inputs_as_uint8:
                sample_in = (255 * sample_in).to(dtype=torch.uint8)
            numpy.savez(sample_input_filename, sample_in)

            # Save outputs as numpy array
            sample_output_filename = sample_out_dir / f"out-{file_idx}.npz"
            numpy.savez(sample_output_filename, *sample_out)
            exported_samples += 1

            if exported_samples >= number_export_samples:
                break

        if exported_samples >= number_export_samples:
            break

    if exported_samples < number_export_samples:
        nm_log_console(
            f"Could not export {number_export_samples} samples. Exhausted dataloader "
            f"and exported {exported_samples} samples",
            level="warning",
        )

    nm_log_console(f"Complete export of {number_export_samples} to {save_dir}")


def _graph_has_uint8_inputs(onnx_path: Union[str, Path]) -> bool:
    """
    Load onnx model and check if it's input is type 2 (unit8)
    """
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    return onnx_model.graph.input[0].type.tensor_type.elem_type == 2
