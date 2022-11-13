import torch

from utils.general import check_version

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def generate_anchors(feats, strides, grid_cell_offset=0.5, device='cpu', is_eval=False):
    """Generate anchors from features."""
    anchor_points = []
    stride_tensor = []
    assert feats is not None
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_point = torch.stack([sx, sy], -1).to(torch.float)
        anchor_points.append(anchor_point.view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # bbox
    else:  # xyxy
        return torch.cat((x1y1, x2y2), dim)  # bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
