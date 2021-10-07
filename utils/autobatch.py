# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import colorstr
from utils.torch_utils import de_parallel, profile


def autobatch(model, imgsz=64, fraction=0.8, device='cpu'):
    # Automatically compute optimal batch size to use `fraction` of available CUDA memory
    prefix = colorstr('autobatch: ')
    print(f'\n{prefix} Computing optimal batch size')

    t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3  # (GB)
    r = torch.cuda.memory_reserved(device) / 1024 ** 3  # (GB)
    a = torch.cuda.memory_allocated(device) / 1024 ** 3  # (GB)
    f = t - (r + a)  # free inside reserved
    # f = 15.8
    print(f'\n{prefix} {t:.3g}G total, {r:.3g}G reserved, {a:.3g}G allocated, {f:.3g}G free')

    batch_sizes = [1, 2, 4, 8]
    model = deepcopy(de_parallel(model)).train()
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)
        y = [x[2] for x in y]  # memory [2]
    except Exception as e:
        print()

    p = np.polyfit(batch_sizes, y, deg=1)  # first degree polynomial fit
    f_intercept = int((f - p[0]) / p[1])  # optimal batch size
    return f_intercept


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)

autobatch(model)
