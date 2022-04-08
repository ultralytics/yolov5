# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 benchmarks on all supported export formats

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
    $ python utils/benchmarks.py --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import export
import val
from utils import notebook_init
from utils.general import LOGGER, print_args
from utils.torch_utils import select_device


def run(weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=640,  # inference size (pixels)
        batch_size=1,  # batch size
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        ):
    y, t = [], time.time()
    formats = export.export_formats()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in formats.iterrows():  # index, (name, file, suffix, gpu-capable)
        try:
            if device.type != 'cpu':
                assert gpu, f'{name} inference not supported on GPU'
            if f == '-':
                w = weights  # PyTorch format
            else:
                w = export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # all others
            assert suffix in str(w), 'export failed'
            result = val.run(data, w, batch_size, imgsz, plots=False, device=device, task='benchmark', half=half)
            metrics = result[0]  # metrics (mp, mr, map50, map, *losses(box, obj, cls))
            speeds = result[2]  # times (preprocess, inference, postprocess)
            y.append([name, round(metrics[3], 4), round(speeds[1], 2)])  # mAP, t_inference
        except Exception as e:
            LOGGER.warning(f'WARNING: Benchmark failure for {name}: {e}')
            y.append([name, None, None])  # mAP, t_inference

    # Print results
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=['Format', 'mAP@0.5:0.95', 'Inference time (ms)'])
    LOGGER.info(f'\nBenchmarks complete ({time.time() - t:.2f}s)')
    LOGGER.info(str(py))
    return py


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
