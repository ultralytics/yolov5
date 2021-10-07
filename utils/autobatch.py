# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr


def autobatch(model, imgsz=640, fraction=0.8, device=0):
    # Automatically compute optimal batch size to use `fraction` of available CUDA memory
    prefix = colorstr('autobatch: ')
    print(f'\n{prefix} Computing optimal batch size')

    t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3  # (GB)
    r = torch.cuda.memory_reserved(device) / 1024 ** 3  # (GB)
    a = torch.cuda.memory_allocated(device) / 1024 ** 3  # (GB)
    f = r - a  # free inside reserved

    try:
        batch_sizes = [1, 2, 4, 8]
        print(f'\n{prefix} {t:.3g}G total, {r:.3g}G reserved, {a:.3g}G allocated, {f:.3g}G free')
    except Exception as e:
        print()


    #x, y = zip(*x)
    #p = np.polyfit(x, y)


    return None
