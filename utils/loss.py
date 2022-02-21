# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
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
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

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

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
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
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


def distillation_loss(output_student, output_teacher, num_classes, temperature=3.0):
    """object detection distillation loss

    :param output_student: the output of student model
    :param output_teacher:  the output of teacher model
    :param num_classes:  classes number
    :param batch_size:  batch size
    :return: loss_st * Lambda_ST
    """

    lambda_st = 0.0001
    batch_size = output_student[0].shape[0]
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    output_student = torch.cat([i.view(-1, num_classes + 5) for i in output_student])
    output_teacher = torch.cat([i.view(-1, num_classes + 5) for i in output_teacher])
    log_softmax_student = nn.functional.log_softmax(output_student/temperature, dim=1)
    softmax_teacher = nn.functional.softmax(output_teacher/temperature, dim=1)
    loss_st = criterion_st(log_softmax_student, softmax_teacher) * (temperature * temperature) / batch_size

    return loss_st * lambda_st


def split_distill_loss(student_outs, teacher_outs, weight):
    """
    Add fine grained distillation losses.
    Each loss is composed by distill_reg_loss, distill_cls_loss and distill_obj_loss
    From the paper ‘Object detection at 200 Frames Per Second’
    """
    distill_reg_loss, distill_cls_loss, distill_obj_loss = [], [], []

    teacher_outs = feature_map_nms(teacher_outs)

    for s_out, t_out in zip(student_outs, teacher_outs):
        student_x, student_y, student_w, student_h, student_obj, \
        student_cls = s_out[..., 0], s_out[..., 1], s_out[..., 2], s_out[..., 3], s_out[..., 4], s_out[..., 5:]
        teacher_x, teacher_y, teacher_w, teacher_h, teacher_obj, \
        teacher_cls = t_out[..., 0], t_out[..., 1], t_out[..., 2], t_out[..., 3], t_out[..., 4], t_out[..., 5:]

        distill_reg_loss.append(
            obj_weighted_reg(student_x, student_y, student_w, student_h,
                             teacher_x, teacher_y, teacher_w, teacher_h, teacher_obj))

        distill_cls_loss.append(
            obj_weighted_cls(student_cls, teacher_cls, teacher_obj))

        distill_obj_loss.append(obj_loss(student_obj, teacher_obj))

    distill_reg_loss = sum(distill_reg_loss)
    distill_cls_loss = sum(distill_cls_loss)
    distill_obj_loss = sum(distill_obj_loss)
    loss = (distill_reg_loss + distill_cls_loss + distill_obj_loss) * weight

    return loss


def obj_weighted_reg(sx, sy, sw, sh, tx, ty, tw, th, tobj):
    """distill_reg_loss"""

    loss_x = F.binary_cross_entropy_with_logits(sx, torch.sigmoid(tx))
    loss_y = F.binary_cross_entropy_with_logits(sy, torch.sigmoid(ty))
    loss_w = torch.abs(sw - tw)
    loss_h = torch.abs(sh - th)
    loss = sum([loss_x, loss_y, loss_w, loss_h])
    weighted_loss = torch.mean(loss * torch.sigmoid(tobj))
    return weighted_loss


def obj_weighted_cls(scls, tcls, tobj):
    """distill_cls_loss"""

    loss = F.binary_cross_entropy_with_logits(scls, torch.sigmoid(tcls))
    weighted_loss = torch.mean(torch.mul(loss, torch.sigmoid(tobj)))
    return weighted_loss


def obj_loss(sobj, tobj):
    """distill_obj_loss"""

    obj_mask = (tobj > 0.).float()
    loss = torch.mean(F.binary_cross_entropy_with_logits(sobj, obj_mask))
    return loss


def feature_map_nms(teacher_outs):
    """faature map nms

    :param teacher_outs: the outputs of teacher model
    :return: outputs
    """
    outputs = []
    for teacher_out in teacher_outs:
        for x in range(0, teacher_out.shape[2] - 3 + 1, 3):
            for y in range(0, teacher_out.shape[3] - 3 + 1, 3):
                grid_3x3 = teacher_out[:, :, x:x+3, y:y+3, :]
                grid_max = grid_3x3.reshape(teacher_out.shape[0], -1, teacher_out.shape[-1])
                max_index = grid_max.argmax(dim=1)
                mask = torch.zeros_like(grid_max)
                for i in range(teacher_out.shape[0]):
                    mask[i, max_index[i, 4]] = grid_max[i, max_index[i, 4]]
                teacher_out[:, :, x:x+3, y:y+3, :] = mask.reshape(grid_3x3.shape)

        outputs.append(teacher_out)
    return outputs


def compute_distillation_output_loss(p, t_p, model, d_weight=1):
    t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError("reduction must be mean in distillation mode!")

    DboxLoss = nn.MSELoss(reduction="none")
    DclsLoss = nn.MSELoss(reduction="none")
    DobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        t_pi = t_p[i]
        t_obj_scale = t_pi[..., 4].sigmoid()

        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        t_lbox += torch.mean(DboxLoss(pi[..., :4], t_pi[..., :4]) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, model.nc)
            # t_lcls += torch.mean(c_obj_scale * (pi[..., 5:] - t_pi[..., 5:]) ** 2)
            t_lcls += torch.mean(DclsLoss(pi[..., 5:], t_pi[..., 5:]) * c_obj_scale)

        # t_lobj += torch.mean(t_obj_scale * (pi[..., 4] - t_pi[..., 4]) ** 2)
        t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['box']
    t_lobj *= h['obj']
    t_lcls *= h['cls']
    # bs = p[0].shape[0]  # batch size
    loss = (t_lobj + t_lbox + t_lcls) * d_weight
    return loss


def compute_distillation_feature_loss(s_f, t_f, model, f_weight=0.1):
    """
    Feature Map distillation.
    Args:
        s_f: student feature
        t_f: teacher feature
        model: model

    Returns: distillation feature loss
    """
    h = model.hyp  # hyperparameters
    ft = torch.cuda.FloatTensor if s_f[0].is_cuda else torch.Tensor
    dl_1, dl_2, dl_3 = ft([0]), ft([0]), ft([0])

    loss_func1 = nn.MSELoss(reduction="mean")
    loss_func2 = nn.MSELoss(reduction="mean")
    loss_func3 = nn.MSELoss(reduction="mean")

    dl_1 += loss_func1(s_f[0], t_f[0])
    dl_2 += loss_func2(s_f[1], t_f[1])
    dl_3 += loss_func3(s_f[2], t_f[2])

    # bs = s_f[0].shape[0]

    return (dl_1 + dl_2 + dl_3) * f_weight