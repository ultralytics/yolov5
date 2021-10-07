# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import colorstr
from utils.torch_utils import de_parallel, profile


def autobatch(model, imgsz=640, fraction=0.95):
    # Automatically compute optimal batch size to use `fraction` of available CUDA memory
    prefix = colorstr('autobatch: ')
    print(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')

    device = next(model.parameters()).device  # get model device
    t = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3  # (GB)
    r = torch.cuda.memory_reserved(device) / 1024 ** 3  # (GB)
    a = torch.cuda.memory_allocated(device) / 1024 ** 3  # (GB)
    f = t - (r + a)  # free inside reserved
    print(f'{prefix}{t:.3g}G total, {r:.3g}G reserved, {a:.3g}G allocated, {f:.3g}G free')

    batch_sizes = [1, 2, 4, 8, 16]
    model = deepcopy(de_parallel(model)).train()
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)
    except Exception as e:
        print(f'{prefix}{e}')

    y = [x[2] for x in y if x]  # memory [2]
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)  # first degree polynomial fit
    f_intercept = int((f * fraction - p[1]) / p[0])  # optimal batch size
    return f_intercept

# autobatch(torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False))
