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


# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
# parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# opt = parser.parse_args()
# opt.cfg = check_file(opt.cfg) 
# device = select_device(opt.device)
# print(opt.cfg)

# model = Model(opt.cfg).to(device)
# print(model)

names= ['Apple','Banana','Pitaya','Snow Pear','Cherry Fruit','Kiwi Fruit','Green Mango','Grape',
                  'Corn','green cabbage','purple cabbage','fresh cut purple cabbage','cauliflower',
                  'broccoli','tomato','bebe pumpkin','golden pumpkin','green pepper' ,
                  'Green Bell Pepper','Red Bell Pepper','Yellow Bell Pepper','Eggplant',
                  'Zucchini','Okra','Carrot','quail egg','papaya','fresh cut papaya','spinach','lettuce',
                  'cole','cantaloupe','Fresh cut cantaloupe','Pleurotus ostreatus','green radish',
                  'baby cabbage','egg','cucumber','yellow mango','green grape','blueberry',
                  'Strawberry','longan','hawthorn','red cherry','juiced peach','nectarine','passion fruit',
                  'Plum','avocado','mangosteen','orange','yellow orange','lemon','yuzu','common pear','bergamot pear',
                  'Bean sprouts','oiled lettuce','celery']

print(len(names),names[56],names[55])