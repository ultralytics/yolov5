import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ..general import xywh2xyxy
from ..metrics import box_iou


def non_max_suppression_masks(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        mask_dim=32,
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (0 <= conf_thres <= 1), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (0 <= iou_thres <= 1), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    nm = 5 + mask_dim

    t = time.time()
    output = [torch.zeros((0, 6 + mask_dim), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        pred_masks = x[:, 5:nm]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, nm:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + nm, None], j[:, None].float(), pred_masks[i]), 1)
        else:  # best class only
            conf, j = x[:, nm:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), pred_masks), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def crop(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = (
        boxes[:, 1],
        boxes[:, 3],
    )

    rows = (torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n))
    cols = (torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n))

    # (1, w, 1), (1, 1, n) -> (1, w, n)
    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    # (h, 1, 1), (1, 1, n) -> (h, 1, n)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    # (h, w, n)
    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def process_mask_upsample(proto_out, out_masks, bboxes, shape):
    """
    Crop after unsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """
    # mask_h, mask_w, n
    masks = proto_out.float().permute(1, 2, 0).contiguous() @ out_masks.float().tanh().T
    masks = masks.sigmoid()
    masks = masks.permute(2, 0, 1).contiguous()
    # [n, mask_h, mask_w]
    masks = F.interpolate(masks.unsqueeze(0), shape, mode='bilinear', align_corners=False).squeeze(0)
    # [mask_h, mask_w, n]
    masks = crop(masks.permute(1, 2, 0).contiguous(), bboxes)
    return masks.gt_(0.5)  # .gt_(0.2)


def process_mask(proto_out, out_masks, bboxes, shape, upsample=False):
    """
    Crop before unsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """
    downsampled_bboxes = bboxes.clone()
    mh, mw = proto_out.shape[1:]
    ih, iw = shape
    # mask_h, mask_w, n
    masks = proto_out.float().permute(1, 2, 0).contiguous() @ out_masks.float().tanh().T
    # print(masks)
    masks = masks.sigmoid()
    # print('after sigmoid:', masks)
    downsampled_bboxes[:, 0] = downsampled_bboxes[:, 0] / iw * mw
    downsampled_bboxes[:, 2] = downsampled_bboxes[:, 2] / iw * mw
    downsampled_bboxes[:, 1] = downsampled_bboxes[:, 1] / ih * mh
    downsampled_bboxes[:, 3] = downsampled_bboxes[:, 3] / ih * mh
    masks = crop(masks, downsampled_bboxes)
    masks = masks.permute(2, 0, 1).contiguous()
    # [n, mask_h, mask_w]
    if upsample:
        masks = F.interpolate(masks.unsqueeze(0), shape, mode='bilinear', align_corners=False).squeeze(0)
    return masks.gt_(0.5).permute(1, 2, 0).contiguous()


def scale_masks(img1_shape, masks, img0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    resize for the most time
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    tl_pad = int(pad[1]), int(pad[0])  # y, x
    br_pad = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    # masks_h, masks_w, n
    masks = masks[tl_pad[0]:br_pad[0], tl_pad[1]:br_pad[1]]
    # 1, n, masks_h, masks_w
    # masks = masks.permute(2, 0, 1).contiguous()[None, :]
    # # shape = [1, n, masks_h, masks_w] after F.interpolate, so take first element
    # masks = F.interpolate(masks, img0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    # masks_h, masks_w, n
    masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]))

    # keepdim
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def mask_iou(mask1, mask2):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, [N, M]
    """
    # print(mask1.shape)
    # print(mask2.shape)
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    return intersection / (union + 1e-7)


def masks_iou(mask1, mask2):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [N, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, (N, )
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1 + area2) - intersection
    return intersection / (union + 1e-7)
