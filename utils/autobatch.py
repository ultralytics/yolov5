# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Auto-batch utils."""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    """
    Checks and computes optimal training batch size for a YOLOv5 model given image size and AMP setting.

    Args:
        model (torch.nn.Module): The YOLOv5 model for which the batch size is being checked.
        imgsz (int, optional): Size of input images for the model. Default is 640.
        amp (bool, optional): Automatic Mixed Precision (AMP) setting. If True, enables AMP. Default is True.

    Returns:
        int: Optimal training batch size for the given model and settings.

    Notes:
        This function profiles the memory usage of the model with a single image to estimate the maximum batch size
        that fits into available GPU memory. It utilizes PyTorch's `torch.cuda.amp.autocast` for AMP if enabled.

    Example:
        ```python
        import torch
        from yolov5 import check_train_batch_size

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        optimal_batch_size = check_train_batch_size(model, imgsz=640, amp=True)
        print(f'Optimal Batch Size: {optimal_batch_size}')
        ```

    For more information, see https://github.com/ultralytics/yolov5.
    """
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    """
    Estimates the optimal batch size for a YOLOv5 model by computing based on specified CUDA memory fraction.

    Args:
        model (torch.nn.Module): The YOLOv5 model instance for which the optimal batch size is to be estimated.
        imgsz (int, optional): Image size (pixels) to be used for the batch size computation. Defaults to 640.
        fraction (float, optional): Fraction of available CUDA memory to use for estimating the batch size. Defaults to 0.8.
        batch_size (int, optional): Default batch size to fall back to in case of computation issues. Defaults to 16.

    Returns:
        int: Estimated optimal batch size based on CUDA memory availability.

    Notes:
        - The function will raise warnings and revert to the default values in scenarios of computation anomalies or
          when CUDA is unavailable.
        - It is advisable to restart the environment and retry if a CUDA anomaly is detected.

    Example:
        ```python
        import torch
        from utils.autobatch import autobatch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
        optimal_batch_size = autobatch(model)
        print(optimal_batch_size)
        ```

        1. Make sure that CUDA is available on your device if you intend to use GPU for the batch size estimation.
        2. The function is sensitive to the state of `torch.backends.cudnn.benchmark`, ensure it is set to `False` to avoid
           fallback to default batch size due to benchmark restrictions.
    """
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")
    device = next(model.parameters()).device  # get model device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f"{prefix}{e}")

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f"{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.")

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
    return b
