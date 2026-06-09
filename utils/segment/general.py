# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Segmentation mask utils."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_mask(masks, boxes):
    """Crop predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        masks (torch.Tensor): Masks with shape (n, h, w).
        boxes (torch.Tensor): Box coordinates with shape (n, 4) in xyxy pixel format (mask resolution).
    """
    _n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """Crop mask prototypes with the predicted boxes, then optionally upsample to the input image size.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (n, mask_dim), n is number of masks after NMS.
        bboxes (torch.Tensor): Box coordinates with shape (n, 4).
        shape (tuple): Input image size as (h, w).
        upsample (bool): Whether to upsample the masks to the input image size.

    Returns:
        (torch.Tensor): Binary masks with shape (n, mask_h, mask_w), upsampled to (n, h, w) when upsample=True.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """Crop mask prototypes to the input image size (native), upsampling and then cropping by the predicted boxes.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (n, mask_dim), n is number of masks after NMS.
        bboxes (torch.Tensor): Box coordinates with shape (n, 4).
        shape (tuple): Input image size as (h, w).

    Returns:
        (torch.Tensor): Binary masks with shape (n, h, w).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """Rescale masks from the model input shape (im1_shape) to the original image shape (im0_shape).

    Args:
        im1_shape (tuple): Model input shape as (h, w).
        masks (np.ndarray): Masks with shape (h, w, num).
        im0_shape (tuple): Original image shape as (h, w, 3).
        ratio_pad (tuple, optional): Ratio and padding for scaling. If None, calculated from the shapes.

    Returns:
        (np.ndarray): Rescaled masks resized to im0_shape.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def mask_iou(mask1, mask2, eps=1e-7):
    """Compute the IoU between two sets of flattened masks.

    Args:
        mask1 (torch.Tensor): Predicted masks with shape (N, n), where n = image_w * image_h.
        mask2 (torch.Tensor): Ground-truth masks with shape (M, n).
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU matrix with shape (N, M).
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks2segments(masks, strategy="largest"):
    """Converts binary (n,160,160) masks to polygon segments with options for concatenation or selecting the largest
    segment.
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments
