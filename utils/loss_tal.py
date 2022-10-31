# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from utils.tal.assigner import TaskAlignedAssigner
from utils.tal.anchor_generator import dist2bbox, generate_anchors, bbox2dist


def xywh2xyxy(x):
    # from utils.general import xywh2xyxy
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape([-1, 4])
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum
        # loss_iou = loss_iou.mean()

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, use_dfl=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=1.0, beta=6.0)
        self.bbox_loss = BboxLoss(16, use_dfl=use_dfl).to(device)
        self.reg_max = 16 if use_dfl else 0
        self.use_dfl = use_dfl
        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1).to(device)

    def preprocess(self, targets, batch_size, scale_tensor):
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[:, :, 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, _ = pred_dist.shape
            # pred_dist = pred_dist.view(b, a, 4, self.reg_max + 1).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, self.reg_max + 1, 4).softmax(2).mul(self.proj.type(pred_dist.dtype).view(1,1,17,1))).sum(2)
            pred_dist = pred_dist.view(b, a, self.reg_max + 1, 4).permute(0, 1, 3, 2).contiguous().softmax(3).matmul(
                self.proj.type(pred_dist.dtype))

        return dist2bbox(pred_dist, anchor_points, box_format="xyxy")

    def __call__(self, p, targets, img=None, epoch=0):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldfl = torch.zeros(1, device=self.device)  # object loss

        feats, pred_obj, pred_scores, pred_distri = p

        # TODO adjust TAL/DFL loss for channel dim=1
        pred_obj = pred_obj.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(feats, torch.tensor([8, 16, 32]), 5.0,
                                                                                 0.5, device=self.device)

        gt_bboxes_scale = torch.full((1, 4), 640).type_as(pred_scores)
        batch_size, grid_size = pred_scores.shape[:2]

        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            (pred_scores if self.nc > 1 else pred_obj).detach().sigmoid(),
            pred_bboxes.detach() * stride_tensor,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt)

        pred_obj = pred_obj.view(batch_size, grid_size)
        tobj = torch.zeros_like(pred_obj)

        target_bboxes /= stride_tensor

        target_scores_sum = target_scores.sum()

        # cls loss
        # target_labels = F.one_hot(target_labels, self.nc)  # (b, h*w, 80)
        # lcls = self.BCEcls(pred_scores[fg_mask], target_scores[fg_mask].to(pred_scores.dtype))  # BCE
        # target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.nc))
        # target_labels = F.one_hot(target_labels.long(), self.nc + 1)[..., :-1]
        lcls = self.BCEcls(pred_scores, target_scores.to(pred_scores.dtype)).sum()  # BCE

        # VFL way
        # lcls = self.varifocal_loss(pred_scores, target_scores, target_labels)
        lcls /= target_scores_sum

        num_pos = fg_mask.sum()

        if num_pos:
            # lcls = self.BCEcls(pred_scores[fg_mask], target_scores[fg_mask].to(pred_scores.dtype)).mean()  # BCE
            # bbox loss
            lbox, ldfl, iou = self.bbox_loss(pred_distri,
                                             pred_bboxes,
                                             anchor_points_s,
                                             target_bboxes,
                                             target_scores,
                                             target_scores_sum,
                                             fg_mask)

            # obj loss
            # tobj[fg_mask] = iou.detach().clamp(0).type(tobj.dtype).squeeze()
            # tobj[fg_mask] = target_scores[fg_mask].detach().clamp(0).type(tobj.dtype).max(1)[0]
            tobj[fg_mask] = 1

            lobj = self.BCEobj(pred_obj, tobj)
            # lobj = 0

        # lbox *= self.hyp["box"] * 3
        lbox *= 2.5 * 3
        lobj *= 0.7 * 3
        lcls *= 1.0 * 3
        ldfl *= 0.5 * 3
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + ldfl) * bs, torch.as_tensor([lbox, ldfl, lcls], device=lbox.device).detach()
