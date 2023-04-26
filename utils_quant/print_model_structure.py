"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.torch_utils import select_device

from pytorch_quantization import nn as quant_nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/finetune/yolov5s-max-512.pth', help='weights path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    # To use Pytorch's own fake quantization functions
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model

    # Print model name and params
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
        print(m)


