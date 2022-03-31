# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmarking script for YOLO ONNX models with the DeepSparse engine.


##########
Command help:
usage: benchmark.py [-h] [-e {deepsparse,onnxruntime,torch}]
                    [--data-path DATA_PATH]
                    [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                    [-b BATCH_SIZE] [-c NUM_CORES]
                    [-i NUM_ITERATIONS] [-w NUM_WARMUP_ITERATIONS] [-q]
                    [--fp16] [--device DEVICE]
                    [--model-config MODEL_CONFIG]
                    model_filepath

Benchmark sparsified YOLO models

positional arguments:
  model_filepath        The full filepath of the ONNX model file or SparseZoo
                        stub to the model for deepsparse and onnxruntime
                        benchmarks. Path to a .pt loadable PyTorch Module for
                        torch benchmarks - the Module can be the top-level
                        object loaded or loaded into 'model' in a state dict

optional arguments:
  -h, --help            show this help message and exit
  -e {deepsparse,onnxruntime,torch}, --engine {deepsparse,onnxruntime,torch}
                        Inference engine backend to run benchmark on. Choices
                        are 'deepsparse', 'onnxruntime', and 'torch'. Default
                        is 'deepsparse'
  --data-path DATA_PATH
                        Optional filepath to image examples to run the
                        benchmark on. Can be path to directory, single image
                        jpg file, or a glob path. All files should be in jpg
                        format. If not provided, sample COCO images will be
                        downloaded from the SparseZoo
  --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to benchmark with, must be two integers.
                        Default is 640 640
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the benchmark for
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the benchmark on,
                        defaults to None where it uses all physical cores
                        available on the system. For DeepSparse benchmarks,
                        this value is the number of cores per socket
  -i NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        The number of iterations the benchmark will be run for
  -w NUM_WARMUP_ITERATIONS, --num-warmup-iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations that will be executed
                        before the actual benchmarking
  -q, --quantized-inputs
                        Set flag to execute benchmark with int8 inputs instead
                        of float32
  --fp16                Set flag to execute torch benchmark in half precision
                        (fp16)
  --device DEVICE       Torch device id to benchmark the model with. Default
                        is cpu. Non cpu benchmarking only supported for torch
                        benchmarking. Default is 'cpu' unless running a torch
                        benchmark and cuda is available, then cuda on device
                        0. i.e. 'cuda', 'cpu', 0, 'cuda:1'
  --model-config MODEL_CONFIG
                        YOLO config YAML file to override default anchor
                        points when post-processing. Defaults to use standard
                        YOLOv3/YOLOv5 anchors

##########
Example command for running a benchmark on a pruned quantized YOLOv5s:
python benchmark.py \
    zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --batch-size 32 \
    --quantized-inputs

##########
Example for benchmarking on a local YOLOv3 PyTorch model on GPU with half precision:
python benchmark.py \
    /PATH/TO/yolov3-spp.pt \
    --engine torch \
    --batch-size 32 \
    --device cuda:0 \
    --half-precision

##########
Example for benchmarking on a local YOLOv5l ONNX with onnxruntime:
python benchmark.py \
    /PATH/TO/yolov5l.onnx \
    --engine onnxruntime \
    --batch-size 32 \

#########
Full list of SparseZoo stubs for benchmarking
* Baseline dense YOLOv3 -
    "zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none"
* Pruned YOLOv3 (87% sparse) -
    "zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive-97"
* Pruned-Quantized YOLOv3 (83% sparse, CPU must support VNNI) -
    "zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive-94"
* Baseline dense YOLOv5l -
    "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none"
* Pruned YOLOv5l (86.3% sparse) -
    "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
* Pruned-Quantized YOLOv5l (79.6% sparse, CPU must support VNNI) -
    "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95"
* Baseline dense YOLOv5s -
    "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none"
* Pruned YOLOv5s (75.6% sparse) -
    "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
* Pruned-Quantized YOLOv5s (68.2% sparse, CPU must support VNNI) -
    "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94"
"""


import argparse
import glob
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy
import onnx
import onnxruntime
from tqdm.auto import tqdm

import torch
from deepsparse import compile_model
from deepsparse.benchmark import BenchmarkResults
from deepsparse_utils import (
    YoloPostprocessor,
    download_pytorch_model_if_stub,
    load_image,
    modify_yolo_onnx_input_shape,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)
from sparseml.onnx.utils import override_model_batch_size
from sparsezoo.models.detection import yolo_v3 as zoo_yolo_v3
from sparsezoo.utils import load_numpy_list


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Benchmark sparsified YOLO models")

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full filepath of the ONNX model file or SparseZoo stub to the model "
            "for deepsparse and onnxruntime benchmarks. Path to a .pt loadable PyTorch "
            "Module for torch benchmarks - the Module can be the top-level object "
            "loaded or loaded into 'model' in a state dict"
        ),
    )

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE],
        help=(
            "Inference engine backend to run benchmark on. Choices are 'deepsparse', "
            "'onnxruntime', and 'torch'. Default is 'deepsparse'"
        ),
    )

    parser.add_argument(
        "--data-path",
        type=Optional[str],
        default=None,
        help=(
            "Optional filepath to image examples to run the benchmark on. Can be path "
            "to directory, single image jpg file, or a glob path. All files should be "
            "in jpg format. If not provided, sample COCO images will be downloaded "
            "from the SparseZoo"
        ),
    )

    parser.add_argument(
        "--image-shape",
        type=int,
        default=(640, 640),
        nargs="+",
        help="Image shape to benchmark with, must be two integers. Default is 640 640",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to run the benchmark for",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the benchmark on, "
            "defaults to None where it uses all physical cores available on the system."
            " For DeepSparse benchmarks, this value is the number of cores per socket"
        ),
    )
    parser.add_argument(
        "-i",
        "--num-iterations",
        help="The number of iterations the benchmark will be run for",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-w",
        "--num-warmup-iterations",
        help=(
            "The number of warmup iterations that will be executed before the actual"
            " benchmarking"
        ),
        type=int,
        default=25,
    )
    parser.add_argument(
        "-q",
        "--quantized-inputs",
        help=("Set flag to execute benchmark with int8 inputs instead of float32"),
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        help=("Set flag to execute torch benchmark in half precision (fp16)"),
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=None,
        help=(
            "Torch device id to benchmark the model with. Default is cpu. Non cpu "
            "benchmarking only supported for torch benchmarking. Default is 'cpu' "
            "unless running a torch benchmark and cuda is available, then cuda on "
            "device 0. i.e. 'cuda', 'cpu', 0, 'cuda:1'"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "YOLO config YAML file to override default anchor points when "
            "post-processing. Defaults to use standard YOLOv3/YOLOv5 anchors"
        ),
    )

    args = parser.parse_args(args=arguments)
    if args.engine == TORCH_ENGINE and args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return args


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except Exception:
        return device


def load_images(
    dataset_path: Optional[str], image_size: Tuple[int]
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
    """
    :param dataset_path: optional path to image files to load, if None, images
        are loaded from the SparseZoo
    :param image_size: size to resize images to
    :return: List of loaded images resized and transposed to given size and list
        of non resized images
    """
    path = str(Path(dataset_path).absolute()) if dataset_path else None

    if not path:  # load from SparseZoo
        zoo_model = zoo_yolo_v3()
        images = load_numpy_list(zoo_model.data_originals.downloaded_path())
        # unwrap npz dict
        key = list(images[0].keys())[0]
        images = [image[key] for image in images]
    elif "*" in path:  # load from local file(s) adapted from yolov5/utils/datasets.py
        images = sorted(glob.glob(path, recursive=True))  # glob
    elif os.path.isdir(path):
        images = sorted(glob.glob(os.path.join(path, "*.*")))  # dir
    elif os.path.isfile(path):
        images = [path]  # files
    else:
        raise Exception(f"ERROR: {path} does not exist")

    numpy.random.shuffle(images)
    model_images = []
    original_images = []
    for image in images:
        model_image, original_image = load_image(image, image_size)
        model_images.append(model_image)
        original_images.append(original_image)
    return model_images, original_images


def _load_model(args) -> (Any, bool):
    # validation
    if args.device not in [None, "cpu"] and args.engine != TORCH_ENGINE:
        raise ValueError(f"device {args.device} is not supported for {args.engine}")
    if args.fp16 and args.engine != TORCH_ENGINE:
        raise ValueError(f"half precision is not supported for {args.engine}")
    if args.quantized_inputs and args.engine == TORCH_ENGINE:
        raise ValueError(f"quantized inputs not supported for {args.engine}")
    if args.num_cores is not None and args.engine == TORCH_ENGINE:
        raise ValueError(
            f"overriding default num_cores not supported for {args.engine}"
        )
    if (
        args.num_cores is not None
        and args.engine == ORT_ENGINE
        and onnxruntime.__version__ < "1.7"
    ):
        raise ValueError(
            "overriding default num_cores not supported for onnxruntime < 1.7.0. "
            "If using an older build with OpenMP, try setting the OMP_NUM_THREADS "
            "environment variable"
        )

    # scale static ONNX graph to desired image shape
    if args.engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        args.model_filepath, _ = modify_yolo_onnx_input_shape(
            args.model_filepath, args.image_shape
        )
        has_postprocessing = yolo_onnx_has_postprocessing(args.model_filepath)

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {args.model_filepath}")
        model = compile_model(args.model_filepath, args.batch_size, args.num_cores)
        print(f"Engine info: {model}")
        if args.quantized_inputs and not model.cpu_vnni:
            print(
                "WARNING: VNNI instructions not detected, "
                "quantization speedup not well supported"
            )
    elif args.engine == ORT_ENGINE:
        print(f"loading onnxruntime model for {args.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if args.num_cores is not None:
            sess_options.intra_op_num_threads = args.num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(args.model_filepath)
        override_model_batch_size(onnx_model, args.batch_size)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )
    elif args.engine == TORCH_ENGINE:
        args.model_filepath = download_pytorch_model_if_stub(args.model_filepath)
        print(f"loading torch model for {args.model_filepath}")
        model = torch.load(args.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(args.device)
        model.eval()
        if args.fp16:
            print("Using half precision")
            model.half()
        else:
            print("Using full precision")
            model.float()
        has_postprocessing = True
    return model, has_postprocessing


def _iter_batches(
    dataset: List[Any],
    batch_size: int,
    iterations: int,
) -> Iterable[Any]:
    iteration = 0
    batch = []
    batch_template = numpy.ascontiguousarray(
        numpy.zeros((batch_size, *dataset[0].shape), dtype=dataset[0].dtype)
    )
    while iteration < iterations:
        for item in dataset:
            batch.append(item)

            if len(batch) == batch_size:
                yield numpy.stack(batch, out=batch_template)

                batch = []
                iteration += 1

                if iteration >= iterations:
                    break


def _preprocess_batch(args, batch: numpy.ndarray) -> Union[numpy.ndarray, torch.Tensor]:
    if args.engine == TORCH_ENGINE:
        batch = torch.from_numpy(batch)
        batch = batch.to(args.device)
        batch = batch.half() if args.fp16 else batch.float()
        batch /= 255.0
    elif not args.quantized_inputs:
        batch = batch.astype(numpy.float32) / 255.0
    return batch


def _run_model(
    args, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    outputs = None
    if args.engine == TORCH_ENGINE:
        outputs = model(batch)
    elif args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def benchmark_yolo(args):
    model, has_postprocessing = _load_model(args)
    print("Loading dataset")
    dataset, _ = load_images(args.data_path, tuple(args.image_shape))
    total_iterations = args.num_iterations + args.num_warmup_iterations
    data_loader = _iter_batches(dataset, args.batch_size, total_iterations)

    print(
        (
            f"Running for {args.num_warmup_iterations} warmup iterations "
            f"and {args.num_iterations} benchmarking iterations"
        ),
        flush=True,
    )

    postprocessor = (
        YoloPostprocessor(args.image_shape, args.model_config)
        if not has_postprocessing
        else None
    )

    results = BenchmarkResults()
    progress_bar = tqdm(total=args.num_iterations)

    for iteration, batch in enumerate(data_loader):
        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
        iter_start = time.time()

        # pre-processing
        batch = _preprocess_batch(args, batch)

        # inference
        outputs = _run_model(args, model, batch)

        # post-processing
        if postprocessor:
            outputs = postprocessor.pre_nms_postprocess(outputs)
        else:
            outputs = outputs[0]  # post-processed values stored in first output

        # NMS
        outputs = postprocess_nms(outputs)

        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
        iter_end = time.time()

        if iteration >= args.num_warmup_iterations:
            results.append_batch(
                time_start=iter_start,
                time_end=iter_end,
                batch_size=args.batch_size,
            )
            progress_bar.update(1)

    progress_bar.close()

    print(f"Benchmarking complete. End-to-end results:\n{results}")

    print(f"End-to-end per image time: {results.ms_per_batch / args.batch_size}ms")


def main():
    args = parse_args()
    assert len(args.image_shape) == 2

    benchmark_yolo(args)


if __name__ == "__main__":
    main()