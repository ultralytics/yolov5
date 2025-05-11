# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
    yaml_save,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment


class iOSModel(torch.nn.Module):
    """An iOS-compatible wrapper for YOLOv5 models that normalizes input images based on their dimensions."""

    def __init__(self, model, im):
        """
        Initializes an iOS compatible model with normalization based on image dimensions.

        Args:
            model (torch.nn.Module): The PyTorch model to be adapted for iOS compatibility.
            im (torch.Tensor): An input tensor representing a batch of images with shape (B, C, H, W).

        Returns:
            None: This method does not return any value.

        Notes:
            This initializer configures normalization based on the input image dimensions, which is critical for
            ensuring the model's compatibility and proper functionality on iOS devices. The normalization step
            involves dividing by the image width if the image is square; otherwise, additional conditions might apply.
        """
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)
            # np = model(im)[0].shape[1]  # number of points
            # self.normalize = torch.tensor([1. / w, 1. / h, 1. / w, 1. / h]).expand(np, 4)  # explicit (faster, larger)

    def forward(self, x):
        """
        Run a forward pass on the input tensor, returning class confidences and normalized coordinates.

        Args:
            x (torch.Tensor): Input tensor containing the image data with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Concatenated tensor with normalized coordinates (xywh), confidence scores (conf),
            and class probabilities (cls), having shape (N, 4 + 1 + C), where N is the number of predictions,
            and C is the number of classes.

        Examples:
            ```python
            model = iOSModel(pretrained_model, input_image)
            output = model.forward(torch_input_tensor)
            ```
        """
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


def export_formats():
    r"""
    Returns a DataFrame of supported YOLOv5 model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame
        includes columns for format name, CLI argument suffix, file extension or directory name, and boolean flags
        indicating if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    """
    Log success or failure, execution time, and file size for YOLOv5 model export functions wrapped with @try_export.

    Args:
        inner_func (Callable): The model export function to be wrapped by the decorator.

    Returns:
        Callable: The wrapped function that logs execution details. When executed, this wrapper function returns either:
            - Tuple (str | torch.nn.Module): On success — the file path of the exported model and the model instance.
            - Tuple (None, None): On failure — None values indicating export failure.

    Examples:
        ```python
        @try_export
        def export_onnx(model, filepath):
            # implementation here
            pass

        exported_file, exported_model = export_onnx(yolo_model, 'path/to/save/model.onnx')
        ```

    Notes:
        For additional requirements and model export formats, refer to the
        [Ultralytics YOLOv5 GitHub repository](https://github.com/ultralytics/ultralytics).
    """
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Logs success/failure and execution details of model export functions wrapped with @try_export decorator."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):
    """
    Export a YOLOv5 model to the TorchScript format.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): Example input tensor to be used for tracing the TorchScript model.
        file (Path): File path where the exported TorchScript model will be saved.
        optimize (bool): If True, applies optimizations for mobile deployment.
        prefix (str): Optional prefix for log messages. Default is 'TorchScript:'.

    Returns:
        (str | None, torch.jit.ScriptModule | None): A tuple containing the file path of the exported model
            (as a string) and the TorchScript model (as a torch.jit.ScriptModule). If the export fails, both elements
            of the tuple will be None.

    Notes:
        - This function uses tracing to create the TorchScript model.
        - Metadata, including the input image shape, model stride, and class names, is saved in an extra file (`config.txt`)
          within the TorchScript model package.
        - For mobile optimization, refer to the PyTorch tutorial: https://pytorch.org/tutorials/recipes/mobile_interpreter.html

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        # Load model
        weights = 'yolov5s.pt'
        device = select_device('')
        model = attempt_load(weights, device=device)

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640).to(device)

        # Export model
        file = Path('yolov5s.torchscript')
        export_torchscript(model, im, file, optimize=False)
        ```
    """
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")

    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    """
    Export a YOLOv5 model to ONNX format with dynamic axes support and optional model simplification.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): A sample input tensor for model tracing, usually the shape is (1, 3, height, width).
        file (pathlib.Path | str): The output file path where the ONNX model will be saved.
        opset (int): The ONNX opset version to use for export.
        dynamic (bool): If True, enables dynamic axes for batch, height, and width dimensions.
        simplify (bool): If True, applies ONNX model simplification for optimization.
        prefix (str): A prefix string for logging messages, defaults to 'ONNX:'.

    Returns:
        tuple[pathlib.Path | str, None]: The path to the saved ONNX model file and None (consistent with decorator).

    Raises:
        ImportError: If required libraries for export (e.g., 'onnx', 'onnx-simplifier') are not installed.
        AssertionError: If the simplification check fails.

    Notes:
        The required packages for this function can be installed via:
        ```
        pip install onnx onnx-simplifier onnxruntime onnxruntime-gpu
        ```

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        # Load model
        weights = 'yolov5s.pt'
        device = select_device('')
        model = attempt_load(weights, map_location=device)

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640).to(device)

        # Export model
        file_path = Path('yolov5s.onnx')
        export_onnx(model, im, file_path, opset=12, dynamic=True, simplify=True)
        ```
    """
    check_requirements("onnx>=1.12.0")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))

    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnxslim"))
            import onnxslim

            LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")
    return f, model_onnx


@try_export
def export_openvino(file, metadata, half, int8, data, prefix=colorstr("OpenVINO:")):
    """
    Export a YOLOv5 model to OpenVINO format with optional FP16 and INT8 quantization.

    Args:
        file (Path): Path to the output file where the OpenVINO model will be saved.
        metadata (dict): Dictionary including model metadata such as names and strides.
        half (bool): If True, export the model with FP16 precision.
        int8 (bool): If True, export the model with INT8 quantization.
        data (str): Path to the dataset YAML file required for INT8 quantization.
        prefix (str): Prefix string for logging purposes (default is "OpenVINO:").

    Returns:
        (str, openvino.runtime.Model | None): The OpenVINO model file path and openvino.runtime.Model object if export is
            successful; otherwise, None.

    Notes:
        - Requires `openvino-dev` package version 2023.0 or higher. Install with:
          `$ pip install openvino-dev>=2023.0`
        - For INT8 quantization, also requires `nncf` library version 2.5.0 or higher. Install with:
          `$ pip install nncf>=2.5.0`

    Examples:
        ```python
        from pathlib import Path
        from ultralytics import YOLOv5

        model = YOLOv5('yolov5s.pt')
        export_openvino(Path('yolov5s.onnx'), metadata={'names': model.names, 'stride': model.stride}, half=True,
                        int8=False, data='data.yaml')
        ```

        This will export the YOLOv5 model to OpenVINO with FP16 precision but without INT8 quantization, saving it to
        the specified file path.
    """
    check_requirements("openvino-dev>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    f = str(file).replace(file.suffix, f"_{'int8_' if int8 else ''}openvino_model{os.sep}")
    f_onnx = file.with_suffix(".onnx")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)

    ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework="onnx", compress_to_fp16=half)  # export

    if int8:
        check_requirements("nncf>=2.5.0")  # requires at least version 2.5.0 to use the post-training quantization
        import nncf
        import numpy as np

        from utils.dataloaders import create_dataloader

        def gen_dataloader(yaml_path, task="train", imgsz=640, workers=4):
            """Generates a DataLoader for model training or validation based on the given YAML dataset configuration."""
            data_yaml = check_yaml(yaml_path)
            data = check_dataset(data_yaml)
            dataloader = create_dataloader(
                data[task], imgsz=imgsz, batch_size=1, stride=32, pad=0.5, single_cls=False, rect=False, workers=workers
            )[0]
            return dataloader

        # noqa: F811

        def transform_fn(data_item):
            """
            Quantization transform function.

            Extracts and preprocess input data from dataloader item for quantization.

            Args:
               data_item: Tuple with data item produced by DataLoader during iteration

            Returns:
                input_tensor: Input data for quantization
            """
            assert data_item[0].dtype == torch.uint8, "input image must be uint8 for the quantization preprocessing"

            img = data_item[0].numpy().astype(np.float32)  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            return np.expand_dims(img, 0) if img.ndim == 3 else img

        ds = gen_dataloader(data)
        quantization_dataset = nncf.Dataset(ds, transform_fn)
        ov_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

    ov.serialize(ov_model, f_ov)  # save
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr("PaddlePaddle:")):
    """
    Export a YOLOv5 PyTorch model to PaddlePaddle format using X2Paddle, saving the converted model and metadata.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): Input tensor used for model tracing during export.
        file (pathlib.Path): Path to the source file to be converted.
        metadata (dict): Additional metadata to be saved alongside the model.
        prefix (str): Prefix for logging information.

    Returns:
        tuple (str, None): A tuple where the first element is the path to the saved PaddlePaddle model, and the
        second element is None.

    Examples:
        ```python
        from pathlib import Path
        import torch

        # Assume 'model' is a pre-trained YOLOv5 model and 'im' is an example input tensor
        model = ...  # Load your model here
        im = torch.randn((1, 3, 640, 640))  # Dummy input tensor for tracing
        file = Path("yolov5s.pt")
        metadata = {"stride": 32, "names": ["person", "bicycle", "car", "motorbike"]}

        export_paddle(model=model, im=im, file=file, metadata=metadata)
        ```

    Notes:
        Ensure that `paddlepaddle` and `x2paddle` are installed, as these are required for the export function. You can
        install them via pip:
        ```
        $ pip install paddlepaddle x2paddle
        ```
    """
    check_requirements(("paddlepaddle>=3.0.0", "x2paddle"))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    f = str(file).replace(".pt", f"_paddle_model{os.sep}")

    pytorch2paddle(module=model, save_dir=f, jit_type="trace", input_examples=[im])  # export
    yaml_save(Path(f) / file.with_suffix(".yaml").name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, nms, mlmodel, prefix=colorstr("CoreML:")):
    """
    Export a YOLOv5 model to CoreML format with optional NMS, INT8, and FP16 support.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): Example input tensor to trace the model.
        file (pathlib.Path): Path object where the CoreML model will be saved.
        int8 (bool): Flag indicating whether to use INT8 quantization (default is False).
        half (bool): Flag indicating whether to use FP16 quantization (default is False).
        nms (bool): Flag indicating whether to include Non-Maximum Suppression (default is False).
        mlmodel (bool): Flag indicating whether to export as older *.mlmodel format (default is False).
        prefix (str): Prefix string for logging purposes (default is 'CoreML:').

    Returns:
        tuple[pathlib.Path | None, None]: The path to the saved CoreML model file, or (None, None) if there is an error.

    Notes:
        The exported CoreML model will be saved with a .mlmodel extension.
        Quantization is supported only on macOS.

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.yolo import Model
        model = Model(cfg, ch=3, nc=80)
        im = torch.randn(1, 3, 640, 640)
        file = Path("yolov5s_coreml")
        export_coreml(model, im, file, int8=False, half=False, nms=True, mlmodel=False)
        ```
    """
    check_requirements("coremltools")
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    if mlmodel:
        f = file.with_suffix(".mlmodel")
        convert_to = "neuralnetwork"
        precision = None
    else:
        f = file.with_suffix(".mlpackage")
        convert_to = "mlprogram"
        precision = ct.precision.FLOAT16 if half else ct.precision.FLOAT32
    if nms:
        model = iOSModel(model, im)
    ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
    ct_model = ct.convert(
        ts,
        inputs=[ct.ImageType("image", shape=im.shape, scale=1 / 255, bias=[0, 0, 0])],
        convert_to=convert_to,
        compute_precision=precision,
    )
    bits, mode = (8, "kmeans") if int8 else (16, "linear") if half else (32, None)
    if bits < 32:
        if mlmodel:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning
                )  # suppress numpy==1.20 float warning, fixed in coremltools==7.0
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        elif bits == 8:
            op_config = ct.optimize.coreml.OpPalettizerConfig(mode=mode, nbits=bits, weight_threshold=512)
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            ct_model = ct.optimize.coreml.palettize_weights(ct_model, config)
    ct_model.save(f)
    return f, ct_model


@try_export
def export_engine(
    model, im, file, half, dynamic, simplify, workspace=4, verbose=False, cache="", prefix=colorstr("TensorRT:")
):
    """
    Export a YOLOv5 model to TensorRT engine format, requiring GPU and TensorRT>=7.0.0.

    Args:
        model (torch.nn.Module): YOLOv5 model to be exported.
        im (torch.Tensor): Input tensor of shape (B, C, H, W).
        file (pathlib.Path): Path to save the exported model.
        half (bool): Set to True to export with FP16 precision.
        dynamic (bool): Set to True to enable dynamic input shapes.
        simplify (bool): Set to True to simplify the model during export.
        workspace (int): Workspace size in GB (default is 4).
        verbose (bool): Set to True for verbose logging output.
        cache (str): Path to save the TensorRT timing cache.
        prefix (str): Log message prefix.

    Returns:
        (pathlib.Path, None): Tuple containing the path to the exported model and None.

    Raises:
        AssertionError: If executed on CPU instead of GPU.
        RuntimeError: If there is a failure in parsing the ONNX file.

    Example:
        ```python
        from ultralytics import YOLOv5
        import torch
        from pathlib import Path

        model = YOLOv5('yolov5s.pt')  # Load a pre-trained YOLOv5 model
        input_tensor = torch.randn(1, 3, 640, 640).cuda()  # example input tensor on GPU
        export_path = Path('yolov5s.engine')  # export destination

        export_engine(model.model, input_tensor, export_path, half=True, dynamic=True, simplify=True, workspace=8, verbose=True)
        ```
    """
    assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    if trt.__version__[0] == "7":  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, "8.0.0", hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix(".onnx")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    else:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace * 1 << 30
    if cache:  # enable timing cache
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        buf = Path(cache).read_bytes() if Path(cache).exists() else b""
        timing_cache = config.create_timing_cache(buf)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine, open(f, "wb") as t:
        t.write(engine if is_trt10 else engine.serialize())
    if cache:  # save timing cache
        with open(cache, "wb") as c:
            c.write(config.get_timing_cache().serialize())
    return f, None


@try_export
def export_saved_model(
    model,
    im,
    file,
    dynamic,
    tf_nms=False,
    agnostic_nms=False,
    topk_per_class=100,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    keras=False,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    """
    Export a YOLOv5 model to the TensorFlow SavedModel format, supporting dynamic axes and non-maximum suppression
    (NMS).

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        im (torch.Tensor): Sample input tensor with shape (B, C, H, W) for tracing.
        file (pathlib.Path): File path to save the exported model.
        dynamic (bool): Flag to indicate whether dynamic axes should be used.
        tf_nms (bool, optional): Enable TensorFlow non-maximum suppression (NMS). Default is False.
        agnostic_nms (bool, optional): Enable class-agnostic NMS. Default is False.
        topk_per_class (int, optional): Top K detections per class to keep before applying NMS. Default is 100.
        topk_all (int, optional): Top K detections across all classes to keep before applying NMS. Default is 100.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.45.
        conf_thres (float, optional): Confidence threshold for detections. Default is 0.25.
        keras (bool, optional): Save the model in Keras format if True. Default is False.
        prefix (str, optional): Prefix for logging messages. Default is "TensorFlow SavedModel:".

    Returns:
        tuple[str, tf.keras.Model | None]: A tuple containing the path to the saved model folder and the Keras model instance,
        or None if TensorFlow export fails.

    Notes:
        - The method supports TensorFlow versions up to 2.15.1.
        - TensorFlow NMS may not be supported in older TensorFlow versions.
        - If the TensorFlow version exceeds 2.13.1, it might cause issues when exporting to TFLite.
          Refer to: https://github.com/ultralytics/yolov5/issues/12489

    Example:
        ```python
        model, im = ...  # Initialize your PyTorch model and input tensor
        export_saved_model(model, im, Path("yolov5_saved_model"), dynamic=True)
        ```
    """
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}<=2.15.1")

        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    if tf.__version__ > "2.13.1":
        helper_url = "https://github.com/ultralytics/yolov5/issues/12489"
        LOGGER.info(
            f"WARNING ⚠️ using Tensorflow {tf.__version__} > 2.13.1 might cause issue when exporting the model to tflite {helper_url}"
        )  # handling issue https://github.com/ultralytics/yolov5/issues/12489
    f = str(file).replace(".pt", "_saved_model")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format="tf")
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(
            tfm,
            f,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
            if check_version(tf.__version__, "2.6")
            else tf.saved_model.SaveOptions(),
        )
    return f, keras_model


@try_export
def export_pb(keras_model, file, prefix=colorstr("TensorFlow GraphDef:")):
    """
    Export YOLOv5 model to TensorFlow GraphDef (*.pb) format.

    Args:
        keras_model (tf.keras.Model): The Keras model to be converted.
        file (Path): The output file path where the GraphDef will be saved.
        prefix (str): Optional prefix string; defaults to a colored string indicating TensorFlow GraphDef export status.

    Returns:
        Tuple[Path, None]: The file path where the GraphDef model was saved and a None placeholder.

    Notes:
        For more details, refer to the guide on frozen graphs: https://github.com/leimao/Frozen_Graph_TensorFlow

    Example:
        ```python
        from pathlib import Path
        keras_model = ...  # assume an existing Keras model
        file = Path("model.pb")
        export_pb(keras_model, file)
        ```
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    f = file.with_suffix(".pb")

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
    return f, None


@try_export
def export_tflite(
    keras_model, im, file, int8, per_tensor, data, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")
):
    # YOLOv5 TensorFlow Lite export
    """
    Export a YOLOv5 model to TensorFlow Lite format with optional INT8 quantization and NMS support.

    Args:
        keras_model (tf.keras.Model): The Keras model to be exported.
        im (torch.Tensor): An input image tensor for normalization and model tracing.
        file (Path): The file path to save the TensorFlow Lite model.
        int8 (bool): Enables INT8 quantization if True.
        per_tensor (bool): If True, disables per-channel quantization.
        data (str): Path to the dataset for representative dataset generation in INT8 quantization.
        nms (bool): Enables Non-Maximum Suppression (NMS) if True.
        agnostic_nms (bool): Enables class-agnostic NMS if True.
        prefix (str): Prefix for log messages.

    Returns:
        (str | None, tflite.Model | None): The file path of the exported TFLite model and the TFLite model instance, or None
        if the export failed.

    Example:
        ```python
        from pathlib import Path
        import torch
        import tensorflow as tf

        # Load a Keras model wrapping a YOLOv5 model
        keras_model = tf.keras.models.load_model('path/to/keras_model.h5')

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640)

        # Export the model
        export_tflite(keras_model, im, Path('model.tflite'), int8=True, per_tensor=False, data='data/coco.yaml',
                      nms=True, agnostic_nms=False)
        ```

    Notes:
        - Ensure TensorFlow and TensorFlow Lite dependencies are installed.
        - INT8 quantization requires a representative dataset to achieve optimal accuracy.
        - TensorFlow Lite models are suitable for efficient inference on mobile and edge devices.
    """
    import tensorflow as tf

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace(".pt", "-fp16.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen

        dataset = LoadImages(check_dataset(check_yaml(data))["train"], img_size=imgsz, auto=False)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        if per_tensor:
            converter._experimental_disable_per_channel = True
        f = str(file).replace(".pt", "-int8.tflite")
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr("Edge TPU:")):
    """
    Exports a YOLOv5 model to Edge TPU compatible TFLite format; requires Linux and Edge TPU compiler.

    Args:
        file (Path): Path to the YOLOv5 model file to be exported (.pt format).
        prefix (str, optional): Prefix for logging messages. Defaults to colorstr("Edge TPU:").

    Returns:
        tuple[Path, None]: Path to the exported Edge TPU compatible TFLite model, None.

    Raises:
        AssertionError: If the system is not Linux.
        subprocess.CalledProcessError: If any subprocess call to install or run the Edge TPU compiler fails.

    Notes:
        To use this function, ensure you have the Edge TPU compiler installed on your Linux system. You can find
        installation instructions here: https://coral.ai/docs/edgetpu/compiler/.

    Example:
        ```python
        from pathlib import Path
        file = Path('yolov5s.pt')
        export_edgetpu(file)
        ```
    """
    cmd = "edgetpu_compiler --version"
    help_url = "https://coral.ai/docs/edgetpu/compiler/"
    assert platform.system() == "Linux", f"export only supported on Linux. See {help_url}"
    if subprocess.run(f"{cmd} > /dev/null 2>&1", shell=True).returncode != 0:
        LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
        sudo = subprocess.run("sudo --version >/dev/null", shell=True).returncode == 0  # sudo installed on system
        for c in (
            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
            'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
            "sudo apt-get update",
            "sudo apt-get install edgetpu-compiler",
        ):
            subprocess.run(c if sudo else c.replace("sudo ", ""), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
    f = str(file).replace(".pt", "-int8_edgetpu.tflite")  # Edge TPU model
    f_tfl = str(file).replace(".pt", "-int8.tflite")  # TFLite model

    subprocess.run(
        [
            "edgetpu_compiler",
            "-s",
            "-d",
            "-k",
            "10",
            "--out_dir",
            str(file.parent),
            f_tfl,
        ],
        check=True,
    )
    return f, None


@try_export
def export_tfjs(file, int8, prefix=colorstr("TensorFlow.js:")):
    """
    Convert a YOLOv5 model to TensorFlow.js format with optional uint8 quantization.

    Args:
        file (Path): Path to the YOLOv5 model file to be converted, typically having a ".pt" or ".onnx" extension.
        int8 (bool): If True, applies uint8 quantization during the conversion process.
        prefix (str): Optional prefix for logging messages, default is 'TensorFlow.js:' with color formatting.

    Returns:
        (str, None): Tuple containing the output directory path as a string and None.

    Notes:
        - This function requires the `tensorflowjs` package. Install it using:
          ```shell
          pip install tensorflowjs
          ```
        - The converted TensorFlow.js model will be saved in a directory with the "_web_model" suffix appended to the original file name.
        - The conversion involves running shell commands that invoke the TensorFlow.js converter tool.

    Example:
        ```python
        from pathlib import Path
        file = Path('yolov5.onnx')
        export_tfjs(file, int8=False)
        ```
    """
    check_requirements("tensorflowjs")
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
    f = str(file).replace(".pt", "_web_model")  # js dir
    f_pb = file.with_suffix(".pb")  # *.pb path
    f_json = f"{f}/model.json"  # *.json path

    args = [
        "tensorflowjs_converter",
        "--input_format=tf_frozen_model",
        "--quantize_uint8" if int8 else "",
        "--output_node_names=Identity,Identity_1,Identity_2,Identity_3",
        str(f_pb),
        f,
    ]
    subprocess.run([arg for arg in args if arg], check=True)

    json = Path(f_json).read_text()
    with open(f_json, "w") as j:  # sort JSON Identity_* in ascending order
        subst = re.sub(
            r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}}}',
            r'{"outputs": {"Identity": {"name": "Identity"}, '
            r'"Identity_1": {"name": "Identity_1"}, '
            r'"Identity_2": {"name": "Identity_2"}, '
            r'"Identity_3": {"name": "Identity_3"}}}',
            json,
        )
        j.write(subst)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    """
    Adds metadata to a TensorFlow Lite (TFLite) model file, supporting multiple outputs according to TensorFlow
    guidelines.

    Args:
        file (str): Path to the TFLite model file to which metadata will be added.
        metadata (dict): Metadata information to be added to the model, structured as required by the TFLite metadata schema.
            Common keys include "name", "description", "version", "author", and "license".
        num_outputs (int): Number of output tensors the model has, used to configure the metadata properly.

    Returns:
        None

    Example:
        ```python
        metadata = {
            "name": "yolov5",
            "description": "YOLOv5 object detection model",
            "version": "1.0",
            "author": "Ultralytics",
            "license": "Apache License 2.0"
        }
        add_tflite_metadata("model.tflite", metadata, num_outputs=4)
        ```

    Note:
        TFLite metadata can include information such as model name, version, author, and other relevant details.
        For more details on the structure of the metadata, refer to TensorFlow Lite
        [metadata guidelines](https://ai.google.dev/edge/litert/models/metadata).
    """
    with contextlib.suppress(ImportError):
        # check_requirements('tflite_support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path("/tmp/meta.txt")
        with open(tmp_file, "w") as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()


def pipeline_coreml(model, im, file, names, y, mlmodel, prefix=colorstr("CoreML Pipeline:")):
    """
    Convert a PyTorch YOLOv5 model to CoreML format with Non-Maximum Suppression (NMS), handling different input/output
    shapes, and saving the model.

    Args:
        model (torch.nn.Module): The YOLOv5 PyTorch model to be converted.
        im (torch.Tensor): Example input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        file (Path): Path to save the converted CoreML model.
        names (dict[int, str]): Dictionary mapping class indices to class names.
        y (torch.Tensor): Output tensor from the PyTorch model's forward pass.
        mlmodel (bool): Flag indicating whether to export as older *.mlmodel format (default is False).
        prefix (str): Custom prefix for logging messages.

    Returns:
        (Path): Path to the saved CoreML model (.mlmodel).

    Raises:
        AssertionError: If the number of class names does not match the number of classes in the model.

    Notes:
        - This function requires `coremltools` to be installed.
        - Running this function on a non-macOS environment might not support some features.
        - Flexible input shapes and additional NMS options can be customized within the function.

    Examples:
        ```python
        from pathlib import Path
        import torch

        model = torch.load('yolov5s.pt')  # Load YOLOv5 model
        im = torch.zeros((1, 3, 640, 640))  # Example input tensor

        names = {0: "person", 1: "bicycle", 2: "car", ...}  # Define class names

        y = model(im)  # Perform forward pass to get model output

        output_file = Path('yolov5s.mlmodel')  # Convert to CoreML
        pipeline_coreml(model, im, output_file, names, y)
        ```
    """
    import coremltools as ct
    from PIL import Image

    f = file.with_suffix(".mlmodel") if mlmodel else file.with_suffix(".mlpackage")
    print(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
    batch_size, ch, h, w = list(im.shape)  # BCHW
    t = time.time()

    # YOLOv5 Output shapes
    spec = model.get_spec()
    out0, out1 = iter(spec.description.output)
    if platform.system() == "Darwin":
        img = Image.new("RGB", (w, h))  # img(192 width, 320 height)
        # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
        out = model.predict({"image": img})
        out0_shape, out1_shape = out[out0.name].shape, out[out1.name].shape
    else:  # linux and windows can not run model.predict(), get sizes from pytorch output y
        s = tuple(y[0].shape)
        out0_shape, out1_shape = (s[1], s[2] - 5), (s[1], 4)  # (3780, 80), (3780, 4)

    # Checks
    nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
    na, nc = out0_shape
    # na, nc = out0.type.multiArrayType.shape  # number anchors, classes
    assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

    # Define output shapes (missing)
    out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
    out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
    # spec.neuralNetwork.preprocessing[0].featureName = '0'

    # Flexible input shapes
    # from coremltools.models.neural_network import flexible_shape_utils
    # s = [] # shapes
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
    # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
    # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
    # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
    # r.add_height_range((192, 640))
    # r.add_width_range((192, 640))
    # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

    # Print
    print(spec.description)

    # Model from spec
    weights_dir = None
    weights_dir = None if mlmodel else str(f / "Data/com.apple.CoreML/weights")
    model = ct.models.MLModel(spec, weights_dir=weights_dir)

    # 3. Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = "confidence"
    nms_spec.description.output[1].name = "coordinates"

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name  # 1x507x80
    nms.coordinatesInputFeatureName = out1.name  # 1x507x4
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    # 4. Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=["confidence", "coordinates"],
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.versionString = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.shortDescription = "https://github.com/ultralytics/yolov5"
    pipeline.spec.description.metadata.author = "glenn.jocher@ultralytics.com"
    pipeline.spec.description.metadata.license = "https://github.com/ultralytics/yolov5/blob/master/LICENSE"
    pipeline.spec.description.metadata.userDefined.update(
        {
            "classes": ",".join(names.values()),
            "iou_threshold": str(nms.iouThreshold),
            "confidence_threshold": str(nms.confidenceThreshold),
        }
    )

    # Save the model
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
    model.input_description["image"] = "Input image"
    model.input_description["iouThreshold"] = f"(optional) IOU Threshold override (default: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence Threshold override (default: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
    model.save(f)  # pipelined
    print(f"{prefix} pipeline success ({time.time() - t:.2f}s), saved as {f} ({file_size(f):.1f} MB)")


@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    per_tensor=False,  # TF per tensor quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    cache="",  # TensorRT: timing cache path
    simplify=False,  # ONNX: simplify model
    mlmodel=False,  # CoreML: Export in *.mlmodel format
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    """
    Exports a YOLOv5 model to specified formats including ONNX, TensorRT, CoreML, and TensorFlow.

    Args:
        data (str | Path): Path to the dataset YAML configuration file. Default is 'data/coco128.yaml'.
        weights (str | Path): Path to the pretrained model weights file. Default is 'yolov5s.pt'.
        imgsz (tuple): Image size as (height, width). Default is (640, 640).
        batch_size (int): Batch size for exporting the model. Default is 1.
        device (str): Device to run the export on, e.g., '0' for GPU, 'cpu' for CPU. Default is 'cpu'.
        include (tuple): Formats to include in the export. Default is ('torchscript', 'onnx').
        half (bool): Flag to export model with FP16 half-precision. Default is False.
        inplace (bool): Set the YOLOv5 Detect() module inplace=True. Default is False.
        keras (bool): Flag to use Keras for TensorFlow SavedModel export. Default is False.
        optimize (bool): Optimize TorchScript model for mobile deployment. Default is False.
        int8 (bool): Apply INT8 quantization for CoreML or TensorFlow models. Default is False.
        per_tensor (bool): Apply per tensor quantization for TensorFlow models. Default is False.
        dynamic (bool): Enable dynamic axes for ONNX, TensorFlow, or TensorRT exports. Default is False.
        cache (str): TensorRT timing cache path. Default is an empty string.
        simplify (bool): Simplify the ONNX model during export. Default is False.
        opset (int): ONNX opset version. Default is 12.
        verbose (bool): Enable verbose logging for TensorRT export. Default is False.
        workspace (int): TensorRT workspace size in GB. Default is 4.
        nms (bool): Add non-maximum suppression (NMS) to the TensorFlow model. Default is False.
        agnostic_nms (bool): Add class-agnostic NMS to the TensorFlow model. Default is False.
        topk_per_class (int): Top-K boxes per class to keep for TensorFlow.js NMS. Default is 100.
        topk_all (int): Top-K boxes for all classes to keep for TensorFlow.js NMS. Default is 100.
        iou_thres (float): IoU threshold for NMS. Default is 0.45.
        conf_thres (float): Confidence threshold for NMS. Default is 0.25.
        mlmodel (bool): Flag to use *.mlmodel for CoreML export. Default is False.

    Returns:
        None

    Notes:
        - Model export is based on the specified formats in the 'include' argument.
        - Be cautious of combinations where certain flags are mutually exclusive, such as `--half` and `--dynamic`.

    Example:
        ```python
        run(
            data="data/coco128.yaml",
            weights="yolov5s.pt",
            imgsz=(640, 640),
            batch_size=1,
            device="cpu",
            include=("torchscript", "onnx"),
            half=False,
            inplace=False,
            keras=False,
            optimize=False,
            int8=False,
            per_tensor=False,
            dynamic=False,
            cache="",
            simplify=False,
            opset=12,
            verbose=False,
            mlmodel=False,
            workspace=4,
            nms=False,
            agnostic_nms=False,
            topk_per_class=100,
            topk_all=100,
            iou_thres=0.45,
            conf_thres=0.25,
        )
        ```
    """
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != "cpu" or coreml, "--half only compatible with GPU export, i.e. use --device 0"
        assert not dynamic, "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    ch = next(model.parameters()).size(1)  # require input image channels
    im = torch.zeros(batch_size, ch, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [""] * len(fmts)  # exported filenames
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose, cache)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half, int8, data)
    if coreml:  # CoreML
        f[4], ct_model = export_coreml(model, im, file, int8, half, nms, mlmodel)
        if nms:
            pipeline_coreml(ct_model, im, file, model.names, y, mlmodel)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        assert not tflite or not tfjs, "TFLite and TF.js models must be exported separately, please pass only one type."
        assert not isinstance(model, ClassificationModel), "ClassificationModel export to TF formats not yet supported."
        f[5], s_model = export_saved_model(
            model.cpu(),
            im,
            file,
            dynamic,
            tf_nms=nms or agnostic_nms or tfjs,
            agnostic_nms=agnostic_nms or tfjs,
            topk_per_class=topk_per_class,
            topk_all=topk_all,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            keras=keras,
        )
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6], _ = export_pb(s_model, file)
        if tflite or edgetpu:
            f[7], _ = export_tflite(
                s_model, im, file, int8 or edgetpu, per_tensor, data=data, nms=nms, agnostic_nms=agnostic_nms
            )
            if edgetpu:
                f[8], _ = export_edgetpu(file)
            add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(s_model.outputs))
        if tfjs:
            f[9], _ = export_tfjs(file, int8)
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path("segment" if seg else "classify" if cls else "")
        h = "--half" if half else ""  # --half FP16 inference arg
        s = (
            "# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference"
            if cls
            else "# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference"
            if seg
            else ""
        )
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """
    Parse command-line options for YOLOv5 model export configurations.

    Args:
        known (bool): If True, uses `argparse.ArgumentParser.parse_known_args`; otherwise, uses `argparse.ArgumentParser.parse_args`.
                      Default is False.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.

    Example:
        ```python
        opts = parse_opt()
        print(opts.data)
        print(opts.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--cache", type=str, default="", help="TensorRT: timing cache file path")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--mlmodel", action="store_true", help="CoreML: Export in *.mlmodel format")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run(**vars(opt))  # Execute the run function with parsed options."""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
