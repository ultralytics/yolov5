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


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


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


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class IOUloss:
    """ Calculate IoU loss.
    """

    def __init__(self, box_format="xywh", iou_type="ciou", reduction="none", eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == "xywh":
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
            elif self.box_format == "xyxy":
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif self.box_format == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

        elif self.box_format == "xywh":
            b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
            b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
            b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
            b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
            b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
            b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
                torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == "giou":
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ["diou", "ciou"]:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                    b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == "diou":
                iou = iou - rho2 / c2
            elif self.iou_type == "ciou":
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == "siou":
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss, iou


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, iou_type="giou"):
        super(BboxLoss, self).__init__()
        self.iou_loss = IOUloss(box_format="xyxy", iou_type=iou_type, eps=1e-7)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape([-1, 4])
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        loss_iou, iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos)
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
    sort_obj_iou = False

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
        self.bbox_loss = BboxLoss(16, use_dfl=use_dfl, iou_type="ciou").cuda()
        self.varifocal_loss = VarifocalLoss().cuda()
        self.reg_max = 16 if use_dfl else 0
        self.use_dfl = use_dfl
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for item in targets.cpu().numpy().tolist():
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
            np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
            targets.device
        )
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1) \
                .matmul(self.proj.to(pred_dist.device).to(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, box_format="xyxy")

    def __call__(self, p, targets, img=None, epoch=0):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldfl = torch.zeros(1, device=self.device)  # object loss

        feats, pred_obj, pred_scores, pred_distri = p
        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(feats, torch.tensor([8, 16, 32]), 5.0,
                                                                                 0.5, device=feats[0].device)

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
