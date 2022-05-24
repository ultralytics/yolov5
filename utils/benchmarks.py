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
import typing as t
from pathlib import Path

import pandas as pd
import yaml

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


class ThresholdError(Exception):
    pass


def get_benchmark_threshold_value(data_path: str, model_name: str) -> t.Union[float, int, None]:
    """
    Given the path to the data configurations and target model name,
    retrieve the target benchmark value
    Args:
        data_path: The location of the data configuration file. e.g. data/coco128.yaml
        model_name: The name of the model. E.g. yolov5s, yolov5n, etc

    Returns:
        The target threshold metric value. E.g. 0.5 (mAP)
    """
    with open(data_path) as f:
        dataset_dict = yaml.safe_load(f)
        if 'benchmarks' in dataset_dict and 'mAP' in dataset_dict['benchmarks']:
            LOGGER.info(f'Attempting to find benchmark threshold for: {dataset_dict["download"]}')
            # yolov5s, yolov5n, etc.
            map_dict = dataset_dict['benchmarks']['mAP']
            if model_name not in map_dict:
                raise ValueError(f'Cannot find benchmark threshold for model: {model_name} in {data_path}')

            map_benchmark_threshold = map_dict[model_name]

            if not 0 <= map_benchmark_threshold <= 1:
                raise ValueError('Please specify a mAP between 0 and 1.0')

            return map_benchmark_threshold


def get_unsupported_formats() -> t.Tuple:
    # coreml: Exception: Model prediction is only supported on macOS version 10.13 or later.
    # engine: Requires gpu and docker container with TensorRT dependencies to run
    # tfjs: Conflict with openvino numpy version (openvino < 1.20, tfjs >= 1.20)
    # edgetpu: requires coral board, cloud tpu or some other external tpu
    return 'edgetpu', 'tfjs', 'engine', 'coreml'


def check_if_formats_exist(unsupported_arguments: t.Tuple) -> None:
    """
    Check to see if the formats actually exists under export_formats().
    An error will be thrown if the argument type does not exist
    Args:
        unsupported_arguments: A tuple of unsupported export formats
    """
    export_formats = export.export_formats()
    valid_export_format_arguments = set(export_formats.Argument)
    for unsupported_arg in unsupported_arguments:
        if unsupported_arg not in valid_export_format_arguments:
            raise ValueError(f'Argument: "{unsupported_arg}" is not a valid export format.\n'
                             f'Valid export formats: {", ".join(valid_export_format_arguments)[: -1]}. \n'
                             f'See export.export_formats() for more info.')


def get_benchmark_values(
    name,  # export format name
    f,  # file format
    suffix,  # suffix of file format. E.g. '.pt', '.tflite', etc.
    gpu,  # run on GPU (boolean value)
    weights,  # weights path
    data,  # data path
    imgsz,  # image size: Two-tuple
    half,  # use FP16 half-precision inference
    batch_size,  # batch size
    device,
) -> t.List:
    assert f not in get_unsupported_formats(), f'{name} not supported'
    if device.type != 'cpu':
        assert gpu, f'{name} inference not supported on GPU'

    # Export
    if f == '-':
        w = weights  # PyTorch format
    else:
        w = export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # all others
    assert suffix in str(w), 'export failed'

    # Validate
    result = val.run(data, w, batch_size, imgsz, plots=False, device=device, task='benchmark', half=half)
    metrics = result[0]  # metrics (mp, mr, map50, map, *losses(box, obj, cls))
    speeds = result[2]  # times (preprocess, inference, postprocess)
    mAP, t_inference = round(metrics[3], 4), round(speeds[1], 2)
    return [name, mAP, t_inference]


def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=640,  # inference size (pixels)
        batch_size=1,  # batch size
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        test=False,  # test exports only
        pt_only=False,  # test PyTorch only
        hard_fail=False,  # Raise errors if model fails to export or mAP lower than target threshold
):
    y, t = [], time.time()
    formats = export.export_formats()
    device = select_device(device)
    # Grab benchmark threshold value
    model_name = str(weights).split('/')[-1].split('.')[0]
    map_benchmark_threshold = get_benchmark_threshold_value(str(data), model_name)

    # get unsupported formats and check if they exist under exports.get_exports()
    check_if_formats_exist(get_unsupported_formats())

    for i, (name, f, suffix, gpu) in formats.iterrows():  # index, (name, file, suffix, gpu-capable)
        if hard_fail:
            if f in get_unsupported_formats():
                continue
            # [name, mAP, t_inference]
            benchmarks = get_benchmark_values(name, f, suffix, gpu, weights, data, imgsz, half, batch_size, device)
            y.append(benchmarks)
            name, mAP, t_inference = benchmarks
            if map_benchmark_threshold and mAP < map_benchmark_threshold:
                raise ThresholdError(f'mAP value: {mAP} is below threshold value: {map_benchmark_threshold}')
        else:
            try:
                y.append(get_benchmark_values(name, f, suffix, gpu, weights, data, imgsz, half, batch_size, device))
            except Exception as e:
                LOGGER.warning(f'WARNING: Benchmark failure for {name}: {e}')
                benchmarks = [name, None, None]
                y.append(benchmarks)

        if pt_only and i == 0:
            break  # break after PyTorch

    # Print results
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=['Format', 'mAP@0.5:0.95', 'Inference time (ms)'] if map else ['Format', 'Export', ''])
    LOGGER.info(f'\nBenchmarks complete ({time.time() - t:.2f}s)')
    # pandas dataframe printing fails in CI with the following error
    # ModuleNotFoundError: No module named 'pandas.io.formats.string'
    try:
        LOGGER.info(str(py if map else py.iloc[:, :2]))
    except Exception:
        pretty_formatted_list = '\n'.join(['\t'.join([str(cell) for cell in row]) for row in py.values.tolist()])
        LOGGER.info(pretty_formatted_list)
    return py


def test(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=640,  # inference size (pixels)
        batch_size=1,  # batch size
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        test=False,  # test exports only
        pt_only=False,  # test PyTorch only
        hard_fail=False  # Raise errors if model fails to export or mAP lower than target threshold
):
    y, t = [], time.time()
    formats = export.export_formats()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in formats.iterrows():  # index, (name, file, suffix, gpu-capable)
        try:
            w = weights if f == '-' else \
                export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # weights
            assert suffix in str(w), 'export failed'
            y.append([name, True])
        except Exception:
            y.append([name, False])  # mAP, t_inference

    # Print results
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # print system info
    py = pd.DataFrame(y, columns=['Format', 'Export'])
    LOGGER.info(f'\nExports complete ({time.time() - t:.2f}s)')
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
    parser.add_argument('--test', action='store_true', help='test exports only')
    parser.add_argument('--pt-only', action='store_true', help='test PyTorch only')
    parser.add_argument('--hard-fail',
                        action='store_true',
                        help='Use to raise errors if conditions are met. '
                        'Also asserts that exported model mAP lies above '
                        'user-defined thresholds.')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
