from models.yolo import Model
from PIL import Image
import os
import cv2
import numpy as np
import torch
import torchvision

from thop import profile
# print(torch.__version__)
# print(torchvision.__version__)


model = Model()
# flops,_ = get_model_complexity_info(model,())
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print('flops:', flops)