# Ultralytics YOLOv5 🚀, AGPL-3.0 license

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_mask(masks, boxes):
    """
    Crop predicted masks by zeroing out all areas outside the predicted bounding boxes.

    Args:
        masks (torch.Tensor): A tensor of shape [n, h, w] containing the masks to be cropped.
        boxes (torch.Tensor): A tensor of shape [n, 4] containing bounding box coordinates in relative point form.

    Returns:
        torch.Tensor: A tensor of cropped masks of shape [n, h, w].

    Notes:
        The function is vectorized to optimize performance, and contributions for optimization are credited to Chong.

    Example:
        ```python
        masks = torch.randn(5, 128, 128)  # Example tensor of 5 masks
        boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.4, 0.4, 0.8, 0.8], ..., [0.2, 0.3, 0.6, 0.6]])
        cropped_masks = crop_mask(masks, boxes)
        ```
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Process and crop masks using upsampled prototypes.

    Args:
        protos (torch.Tensor): Tensor of shape (mask_dim, mask_h, mask_w) containing mask prototypes.
        masks_in (torch.Tensor): Tensor of shape (n, mask_dim), where n is the number of masks after NMS.
        bboxes (torch.Tensor): Tensor of shape (n, 4) representing bounding box coordinates for each mask.
        shape (tuple[int, int]): Tuple representing the target input image size (height, width).

    Returns:
        torch.Tensor: Tensor of shape (h, w, n) containing the processed and cropped masks, where n is the number of masks.

    Notes:
        - This function interpolates masks using a bilinear mode and aligns corners.
        - Cropped masks are further processed using the `crop_mask` function.
    """

    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Process and crop segmentation masks, optionally upsampling the masks after cropping.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w] representing the mask prototypes.
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim] containing the masks after non-maximum suppression (NMS).
        bboxes (torch.Tensor): A tensor of shape [n, 4] representing the bounding boxes for each mask after NMS.
        shape (tuple[int, int]): The shape of the input image, given as (height, width).
        upsample (bool): A flag indicating whether to upsample the masks to the original input image size after cropping. Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape (height, width, n) containing the final processed masks, where 'height' and 'width' correspond to the dimensions of the input image and 'n' is the number of masks.

    Notes:
        - This function first crops the masks using the specified bounding boxes, then optionally upsamples the cropped masks to match the input image size if `upsample` is set to True.
        - The bounding boxes are adjusted to match the downsampled dimensions of the prototype masks before cropping.

    Examples:
        ```python
        processed_masks = process_mask(protos, masks_in, bboxes, shape, upsample=True)
        ```

        ```python
        processed_masks = process_mask(protos, masks_in, bboxes, shape, upsample=False)
        ```
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
    """
    rocess_mask_native(protos, masks_in, bboxes, shape) Crop and resize predicted masks after upsampling, accounting for
    bounding boxes and image shapes.

    Args:
        protos (torch.Tensor): Prototype masks of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): Mask predictions of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple[int, int]): Shape of the input image as (height, width).

    Returns:
        torch.Tensor: Processed mask tensor of shape [n, shape[0], shape[1]] with values thresholded at 0.5.

    Notes:
        The function performs cropping before resizing, ensuring that masks are properly aligned with their bounding boxes.

    Example:
        ```python
        protos = torch.rand(32, 160, 160)
        masks_in = torch.rand(10, 32)
        bboxes = torch.rand(10, 4)
        shape = (640, 480)
        processed_masks = process_mask_native(protos, masks_in, bboxes, shape)
        ```
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
    """
    Scale and crop masks to match the original image shape.

    Args:
        im1_shape (tuple[int, int]): Model input shape as a tuple (height, width).
        masks (np.ndarray): Array of shape (height, width, num) containing the masks to be scaled.
        im0_shape (tuple[int, int, int]): Original image shape as a tuple (height, width, channels).
        ratio_pad (tuple[tuple[float, float], tuple[float, float]] | None, optional): Scaling ratio and padding
            values. Defaults to None.

    Returns:
        np.ndarray: Array of shape (height, width, num) containing the scaled masks.

    Raises:
        ValueError: If the shape of `masks` is not a 2D or 3D array.

    Notes:
        - This function rescales the mask coordinates proportionally from the model input shape (`im1_shape`)
          to the original image shape (`im0_shape`). When `ratio_pad` is provided, it directly uses the padding
          values instead of calculating them.
        - Requires OpenCV to perform image resizing.

    Examples:
        Scaled masks for a given model input and original shapes:

        ```python
        im1_shape = (640, 480)
        masks = np.random.rand(640, 480, 3)
        im0_shape = (480, 360, 3)

        scaled_masks = scale_image(im1_shape, masks, im0_shape)
        ```
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
    """
    Calculate Intersection over Union (IoU) for pairs of masks.

    Args:
        mask1 (torch.Tensor): A tensor of shape [N, n] representing N predicted masks, where each mask is flattened
            into a vector of length n (image_w * image_h).
        mask2 (torch.Tensor): A tensor of shape [M, n] representing M ground truth (GT) masks, where each mask is
            flattened into a vector of length n (image_w * image_h).
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape [N, M] containing the IoU values for each pair of predicted and ground
        truth masks.

    Note:
        - IoU is calculated as the intersection area divided by the union area of two masks.

    Example:
        ```python
        mask1 = torch.rand(5, 100*100)  # 5 predicted masks of size 100x100
        mask2 = torch.rand(3, 100*100)  # 3 ground truth masks of size 100x100
        iou = mask_iou(mask1, mask2)
        print(iou.shape)  # Output: torch.Size([5, 3])
        ```

        - Refer to https://github.com/ultralytics/ultralytics for more details and usage examples.
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) for pairs of predicted and ground truth masks.

    Args:
        mask1 (torch.Tensor): Tensor of shape [N, n] representing N predicted masks, where n is the flattened mask size
            (image width × image height).
        mask2 (torch.Tensor): Tensor of shape [N, n] representing N ground truth masks, where n is the flattened mask
            size (image width × image height).
        eps (float, optional): Small epsilon value to avoid division by zero, default is 1e-7.

    Returns:
        torch.Tensor: Tensor of shape (N,) containing IoU values for each pair of predicted and ground truth masks.

    Notes:
        This function calculates the IoU for each pair of masks by computing the intersection and union of the predicted
        and ground truth masks.

    Example:
        ```python
        import torch

        mask1 = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        mask2 = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32)
        iou = masks_iou(mask1, mask2)
        ```

        The resulting `iou` tensor will be:
        ```python
        tensor([0.5, 0.3333])
        ```

    See Also:
        mask_iou: A similar function to compute IoU for differently sized masks.
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks2segments(masks, strategy="largest"):
    """
    Converts binary masks to polygon segments selecting either the largest segment or concatenating all segments.

    Args:
        masks (torch.Tensor): Tensor containing binary masks of shape (n, 160, 160), where 'n' is the number of masks.
        strategy (str): The strategy for handling multiple segments per mask. Options are 'largest' to select the largest
            segment or 'concat' to concatenate all segments. Default is 'largest'.

    Returns:
        list[np.ndarray]: A list of numpy arrays where each array contains the polygon coordinates for each mask segment.

    Notes:
        This function uses OpenCV's `findContours` method to extract polygon segments from binary masks.

    Examples:
        ```python
        masks = torch.randn(5, 160, 160) > 0.5  # Example binary masks
        segments = masks2segments(masks, strategy="largest")
        ```
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
