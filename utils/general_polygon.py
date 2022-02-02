from shapely import geometry, geos

from utils.general import *

try:
    # if error in importing polygon_inter_union_cuda, polygon_b_inter_union_cuda, please cd to ./iou_cuda and run "python setup.py install"
    from polygon_inter_union_cuda import polygon_b_inter_union_cuda, polygon_inter_union_cuda
    polygon_inter_union_cuda_enable = True
    polygon_b_inter_union_cuda_enable = True
except Exception as e:
    print(f'Warning: "polygon_inter_union_cuda" and "polygon_b_inter_union_cuda" are not installed.')
    print(f'The Exception is: {e}.')
    polygon_inter_union_cuda_enable = False
    polygon_b_inter_union_cuda_enable = False
# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------

def xyxyxyxyn2xyxyxyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized xyxyxyxy or segments into pixel xyxyxyxy or segments
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = w * x[:, 0::2] + padw  # all x
    y[:, 1::2] = h * x[:, 1::2] + padh  # all y
    return y


def polygon_segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 polygon box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxyxyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside].reshape(-1, 1), y[inside].reshape(-1, 1)
    multipoint = geometry.MultiPoint(np.concatenate((x, y), axis=1))
    # polygon_box: x1, y1, x2, y2, x3, y3, x4, y4 (unnormalized)
    polygon_box = np.array(multipoint.minimum_rotated_rectangle.exterior.coords[:-1]).ravel()
    polygon_box[0::2] = polygon_box[0::2].clip(0., width)
    polygon_box[1::2] = polygon_box[1::2].clip(0., height)
    return polygon_box if any(x) else np.zeros((1, 8))  # xyxyxyxy


def polygon_segments2boxes(segments, img_shapes=None):
    # Convert segment labels to polygon box labels, i.e. (xy1, xy2, ...) to polygon (xyxyxyxy)
    boxes = []
    img_shapes = [None]*len(segments) if img_shapes is None else img_shapes
    for segment, img_shape in zip(segments, img_shapes):
        polygon_box = polygon_segment2box(segment) if img_shape is None else polygon_segment2box(segment, img_shape[1], img_shape[0])
        boxes.append(polygon_box)  # list with item of xyxyxyxy
    return np.array(boxes)  # numpy array with row of xyxyxyxy


def polygon_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    polygon_clip_coords(coords, img0_shape)  # inplace operation
    return coords


def polygon_clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::2].clamp_(0, img_shape[1])  # x1x2x3x4
    boxes[:, 1::2].clamp_(0, img_shape[0])  # y1y2y3y4


def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """

    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        for j in range(m):
            polygon2 = geometry.Polygon(boxes2[j, :].view(4,2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except geos.TopologicalError:
                    print('geos.TopologicalError occured')
    return inter, union


def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

    if torch.cuda.is_available() and polygon_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.

        inter_nan, union_nan = inter.isnan(), union.isnan()
        if inter_nan.any() or union_nan.any():
            inter2, union2 = polygon_inter_union_cuda(boxes1_, boxes2_)  # Careful that order should be: boxes1_, boxes2_.
            inter2, union2 = inter2.T, union2.T
            inter = torch.where(inter_nan, inter2, inter)
            union = torch.where(union_nan, union2, union)
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                    w1, h1 = b1_x2[i]-b1_x1[i], b1_y2[i]-b1_y1[i]+eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU


def polygon_b_inter_union_cpu(boxes1, boxes2):
    """
        iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py;
        Boxes and Anchors having the same shape: nx8;
        Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
    """

    n = boxes1.shape[0]
    inter = torch.zeros(n,)
    union = torch.zeros(n,)
    for i in range(n):
        polygon1 = geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        polygon2 = geometry.Polygon(boxes2[i, :].view(4,2)).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter[i] = polygon1.intersection(polygon2).area
                union[i] = polygon1.union(polygon2).area
            except geos.TopologicalError:
                print('geos.TopologicalError occured')
    return inter, union


def polygon_bbox_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
    """
        Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

    if torch.cuda.is_available() and polygon_b_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside inter_union_cuda must be torch.cuda.float, not double type or half type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_b_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.

        inter_nan, union_nan = inter.isnan(), union.isnan()
        if inter_nan.any() or union_nan.any():
            inter2, union2 = polygon_b_inter_union_cuda(boxes1_, boxes2_)  # Careful that order should be: boxes1_, boxes2_.
            inter2, union2 = inter2.T, union2.T
            inter = torch.where(inter_nan, inter2, inter)
            union = torch.where(union_nan, union2, union)
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_b_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0
    iou = iou.to(device='cuda')
    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # n,
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # n,
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # n,
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # n,
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                iou -= rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                w1, h1 = b1_x2-b1_x1, b1_y2-b1_y1+eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                iou -= (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            iou -= (c_area - union) / c_area  # GIoU
    return iou  # IoU


def polygon_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results for polygon boxes
        Returns:  list of detections, on (n,10) tensor per image [xyxyxyxy, conf, cls]
    """

    # prediction has the shape of (bs, all potential anchors, 89)
    assert not agnostic, "polygon does not support agnostic"
    nc = prediction.shape[2] - 9  # number of classes
    xc = prediction[..., 8] > conf_thres  # confidence candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 3, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into polygon_nms_kernel, can increase this value
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 10), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # xw = (x[..., 0:8:2].max(dim=-1)[0] - x[..., 0:8:2].min(dim=-1)[0]).view(-1, 1)
        # xh = (x[..., 1:8:2].max(dim=-1)[0] - x[..., 1:8:2].min(dim=-1)[0]).view(-1, 1)
        # x[((xw < min_wh) | (xw > max_wh) | (xh < min_wh) | (xh > max_wh)).any(1), 8] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 9), device=x.device)
            v[:, :8] = l[:, 1:9]  # box
            v[:, 8] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 9] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 9:] *= x[:, 8:9]  # conf = obj_conf * cls_conf

        # Box (x1, y1, x2, y2, x3, y3, x4, y4)
        box = x[:, :8].clone()

        # Detections matrix nx10 (xyxyxyxy, conf, cls)
        # Transfer sigmoid probabilities of classes (e.g. three classes [0.567, 0.907, 0.01]) to selected classes (1.0)
        if multi_label:
            i, j = (x[:, 9:] > conf_thres).nonzero(as_tuple=False).T
            # concat satisfied boxes (multi-label-enabled) along 0 dimension
            x = torch.cat((box[i], x[i, j + 9, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 9:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 9:10] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 8].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Polygon NMS does not support Batch NMS and Agnostic
        # x is the sorted predictions with boxes x[:, :8], confidence x[:, 8], class x[:, 9]
        # cannot use torchvision.ops.nms, which only deals with axis-aligned boxes
        i = polygon_nms_kernel(x, iou_thres)  # polygon-NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            boxes = x[:, :8]
            # update boxes as boxes(i,8) = weights(i,n) * polygon boxes(n,8)
            iou = polygon_box_iou(boxes[i], boxes, device=prediction.device) > iou_thres  # iou matrix
            weights = iou * x[:, 8][None]  # polygon box weights
            x[i, :8] = torch.mm(weights, x[:, :8]).float() / weights.sum(1, keepdim=True)  # merged polygon boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def polygon_nms_kernel(x, iou_thres):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9]
        Return the selected indices
    """

    unique_labels = x[:, 9].unique()
    _, scores_sort_index = torch.sort(x[:, 8], descending=True)
    x = x[scores_sort_index]
    x[:, :8] = order_corners(x[:, :8])
    indices = scores_sort_index
    selected_indices = []

    # Iterate through all predicted classes
    for unique_label in unique_labels:
        x_ = x[x[:, 9]==unique_label]
        indices_ = indices[x[:, 9]==unique_label]

        while x_.shape[0]:
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(x_) == 1: break
            # Compute the IOUs for all other the polygon boxes
            iou = polygon_box_iou(x_[0:1, :8], x_[1:, :8], device=x.device, ordered=True).view(-1)
            # Remove overlapping detections with IoU >= NMS threshold
            x_ = x_[1:][iou < iou_thres]
            indices_ = indices_[1:][iou < iou_thres]

    return torch.LongTensor(selected_indices)


def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions:
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """

    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y) # sort y
    x_sorted = torch.zeros_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        x_sorted[i] = x[i, y_indices[i]]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(y.shape[0]):
        y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()




def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)
