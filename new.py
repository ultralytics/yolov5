from PIL import Image
import os
import cv2
import numpy as np
import torch
import torchvision

# print(torch.__version__)
# print(torchvision.__version__)


s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
# print(s)
pf = '%20s'   # print format
print(pf % ('all'))