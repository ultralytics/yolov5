# 这是一个测试文件，与yolo无关，只是用来 测试一部分函数功能

import argparse
from utils.torch_utils import select_device
from utils.general import check_file
from models.yolo import Model
from PIL import Image
import os
import cv2
import numpy as np
import torch
import torchvision

from thop import profile


# model = Model()
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input, ))
# print('flops:', flops)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='models/yolov5m_mydata.yaml', help='model.yaml')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()
opt.cfg = check_file(opt.cfg) 
device = select_device(opt.device)
print(opt.cfg)

model = Model(opt.cfg).to(device)
# print(model)

