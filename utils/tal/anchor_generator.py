import torch
from utils.general import check_version

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False):
    """Generate anchors from features."""
    anchors = []
    anchor_points = []
    stride_tensor = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor_point = torch.stack([sx, sy], -1).to(torch.float)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        num_anchors_list = []
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            sx = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            sy = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor = torch.stack([sx - cell_half_size, sy - cell_half_size,
                                  sx + cell_half_size, sy + cell_half_size], -1).clone().to(feats[0].dtype)
            anchor_point = torch.stack([sx, sy], -1).clone().to(feats[0].dtype)
            anchors.append(anchor.reshape([-1, 4]))
            anchor_points.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    return torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)  # dist
