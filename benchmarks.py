# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 benchmarks on all supported export formats.

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

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU
    $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  # TensorRT

Usage:
    $ python benchmarks.py --weights yolov5s.pt --img 640
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import export
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from segment.val import run as val_seg
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """
    Run YOLOv5 benchmarks on multiple export formats and log results for model performance evaluation.

    Args:
        weights (Path | str): Path to the model weights file (default: ROOT / "yolov5s.pt").
        imgsz (int): Inference size in pixels (default: 640).
        batch_size (int): Batch size for inference (default: 1).
        data (Path | str): Path to the dataset.yaml file (default: ROOT / "data/coco128.yaml").
        device (str): CUDA device, e.g., '0' or '0,1,2,3' or 'cpu' (default: None).
        half (bool): Use FP16 half-precision inference (default: False).
        test (bool): Test export formats only (default: False).
        pt_only (bool): Test only the PyTorch export format (default: False).
        hard_fail (bool): Raise an error on benchmark failure (default: False).

    Returns:
        pd.DataFrame: DataFrame containing benchmark results with the following columns:
            - Format: Export format name.
            - Size (MB): Model size in megabytes.
            - mAP50-95: Mean average precision across multiple IoU thresholds.
            - Inference time (ms): Time taken for inference in milliseconds.

    Notes:
        - Ensure required dependencies are installed using:
          ```
          $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu
          $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow
          $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
          ```

    Example:
        ```python
        $ python benchmarks.py --weights yolov5s.pt --img 640
        ```
    """
    y, t = [], time.time()
    device = select_device(device)
    model_type = type(attempt_load(weights, fuse=False))  # DetectionModel, SegmentationModel, etc.
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, CPU, GPU)
        try:
            assert i not in (9, 10), "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if f == "-":
                w = weights  # PyTorch format
            else:
                w = export.run(
                    weights=weights, imgsz=[imgsz], include=[f], batch_size=batch_size, device=device, half=half
                )[-1]  # all others
            assert suffix in str(w), "export failed"

            # Validate
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][7]  # (box(p, r, map50, map), mask(p, r, map50, map), *loss(box, obj, cls))
            else:  # DetectionModel:
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][3]  # (p, r, map50, map, *loss(box, obj, cls))
            speed = result[2][1]  # times (preprocess, inference, postprocess)
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  # MB, mAP, t_inference
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f"Benchmark --hard-fail for {name}: {e}"
            LOGGER.warning(f"WARNING ⚠️ Benchmark failure for {name}: {e}")
            y.append([name, None, None, None])  # mAP, t_inference
        if pt_only and i == 0:
            break  # break after PyTorch

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  # print system info
    c = ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"] if map else ["Format", "Export", "", ""]
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f"\nBenchmarks complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py["mAP50-95"].array  # values to compare to floor
        floor = eval(hard_fail)  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"HARD FAIL: mAP50-95 < floor {floor}"
    return py


def test(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    test=False,  # test exports only
    pt_only=False,  # test PyTorch only
    hard_fail=False,  # throw error on benchmark failure
):
    """
    Run YOLOv5 export tests for all supported formats and log the results, including inference speed and mAP.

    Args:
        weights (pathlib.Path | str): Path to the model weights file, typically a .pt file for PyTorch, i.e. 'yolov5s.pt'.
        imgsz (int): Inference size in pixels. Default is 640.
        batch_size (int): Batch size for inference. Default is 1.
        data (pathlib.Path | str): Path to the dataset configuration YAML file. Default is 'data/coco128.yaml'.
        device (str): Device to run inference on, i.e., '0', '0,1,2,3' for CUDA-enabled GPUs or 'cpu' for CPU. Default is ''.
        half (bool): Whether to use FP16 half-precision inference. Default is False.
        test (bool): If True, test exports only without inference. Default is False.
        pt_only (bool): If True, test exports for PyTorch format only. Default is False.
        hard_fail (bool): If True, raise an error on benchmark failure. Default is False.

    Returns:
        pandas.DataFrame: A DataFrame logging the results, including columns for export format, export success, mAP score,
                          and inference time (ms), if available.

    Notes:
        - Ensure all required dependencies are installed as outlined in the repository's [requirements](https://github.com/ultralytics/ultralytics#requirements).
        - Edge TPU and TensorFlow.js export formats are not supported.
        - CoreML inference is only supported on macOS 10.13 or later.
    """
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  # index, (name, file, suffix, gpu-capable)
        try:
            w = (
                weights
                if f == "-"
                else export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]
            )  # weights
            assert suffix in str(w), "export failed"
            y.append([name, True])
        except Exception:
            y.append([name, False])  # mAP, t_inference

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=["Format", "Export"])
    LOGGER.info(f"\nExports complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    return py


def parse_opt():
    """
    Parses command-line arguments for YOLOv5 model inference configuration.

    Args:
        weights (str): The path to the weights file. Defaults to 'ROOT / "yolov5s.pt"'.
        imgsz (int): Inference size in pixels. Defaults to 640.
        batch_size (int): Batch size. Defaults to 1.
        data (str): Path to the dataset YAML file. Defaults to 'ROOT / "data/coco128.yaml"'.
        device (str): CUDA device, e.g., '0' or '0,1,2,3' or 'cpu'. Defaults to an empty string (auto-select).
        half (bool): Use FP16 half-precision inference. Defaults to False.
        test (bool): Test exports only. Defaults to False.
        pt_only (bool): Test PyTorch only. Defaults to False.
        hard_fail (bool | str): Throw an error on benchmark failure or specify a minimum metric floor. Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Notes:
        This function completes by calling 'print_args(vars(opt))' to display the command-line arguments.

    Examples:
        Typical usage examples include:
        ```python
        opt = parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--test", action="store_true", help="test exports only")
    parser.add_argument("--pt-only", action="store_true", help="test PyTorch only")
    parser.add_argument("--hard-fail", nargs="?", const=True, default=False, help="Exception on error or < min metric")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 benchmark tests or main model inference and training routines based on user-defined options.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments that include options for weights, image size, batch size,
                                  dataset configuration, device, and other inference settings. Configurations:
                                  - weights (str): Path to the weights file. Default is 'yolov5s.pt'.
                                  - imgsz (int): Inference image size in pixels. Default is 640.
                                  - batch_size (int): Number of images to process in a batch. Default is 1.
                                  - data (str): Path to the dataset configuration YAML file. Default is 'data/coco128.yaml'.
                                  - device (str): Device to run inference on, e.g., 'cpu' or '0' for GPU. Default is ''.
                                  - half (bool): Use FP16 half-precision inference. Default is False.
                                  - test (bool): If True, run export validation tests only. Default is False.
                                  - pt_only (bool): If True, benchmark PyTorch format only. Default is False.
                                  - hard_fail (bool | str): If True, exceptions on benchmark failures or specify a minimum
                                                             performance threshold for metrics. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing benchmark results with columns for format, size in MB, mAP50-95, and inference
                      time in milliseconds if any benchmarking is done; otherwise, logs other results based on opt settings.

    Notes:
        Supported export formats include PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel,
        TensorFlow GraphDef, TensorFlow Lite, TensorFlow Edge TPU, and TensorFlow.js.
        For more details, refer to the official YOLOv5 repository: https://github.com/ultralytics/ultralytics.

    Example:
        ```
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```
    """
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
