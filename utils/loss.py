# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
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

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        bs = p[0].shape[0]  # batch size
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses
        tcls, tbox, indices = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # tgt obj

            n_labels = b.shape[0]  # number of labels
            if n_labels:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                # pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                # pwh = (0.0 + (pwh - 1.09861).sigmoid() * 4) * anchors[i]
                # pwh = (0.33333 + (pwh - 1.09861).sigmoid() * 2.66667) * anchors[i]
                # pwh = (0.25 + (pwh - 1.38629).sigmoid() * 3.75) * anchors[i]
                # pwh = (0.20 + (pwh - 1.60944).sigmoid() * 4.8) * anchors[i]
                # pwh = (0.16667 + (pwh - 1.79175).sigmoid() * 5.83333) * anchors[i]
                pxy = pxy.sigmoid() * 1.6 - 0.3
                pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                loss[0] += (1.0 - iou).mean()  # box loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, gj, gi, iou = b[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n_labels), tcls[i]] = self.cp
                    loss[2] += self.BCEcls(pcls, t)  # cls loss

            obji = self.BCEobj(pi[:, 4], tobj)
            loss[1] += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / self.anchors[i]  # wh ratio
                j = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
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
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            b, c = bc.long().T  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            tcls.append(c)  # class

        return tcls, tbox, indices


class ComputeLoss_NEW:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
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

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.BCE_base = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, p, targets):  # predictions, targets
        tcls, tbox, indices = self.build_targets(p, targets)  # targets
        bs = p[0].shape[0]  # batch size
        n_labels = targets.shape[0]  # number of labels
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses

        # Compute all losses
        all_loss = []
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            if n_labels:
                pxy, pwh, pobj, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 2)  # target-subset of predictions

                # Regression
                pbox = torch.cat((pxy.sigmoid() * 1.6 - 0.3, (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]), 2)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(predicted_box, target_box)
                obj_target = iou.detach().clamp(0).type(pi.dtype)  # objectness targets

                all_loss.append([(1.0 - iou) * self.hyp['box'],
                                 self.BCE_base(pobj.squeeze(), torch.ones_like(obj_target)) * self.hyp['obj'],
                                 self.BCE_base(pcls, F.one_hot(tcls[i], self.nc).float()).mean(2) * self.hyp['cls'],
                                 obj_target,
                                 tbox[i][..., 2] > 0.0])  # valid

        # Lowest 3 losses per label
        n_assign = 4  # top n matches
        cat_loss = [torch.cat(x, 1) for x in zip(*all_loss)]
        ij = torch.zeros_like(cat_loss[0]).bool()  # top 3 mask
        sum_loss = cat_loss[0] + cat_loss[2]
        for col in torch.argsort(sum_loss, dim=1).T[:n_assign]:
            # ij[range(n_labels), col] = True
            ij[range(n_labels), col] = cat_loss[4][range(n_labels), col]
        loss[0] = cat_loss[0][ij].mean() * self.nl  # box loss
        loss[2] = cat_loss[2][ij].mean() * self.nl  # cls loss

        # Obj loss
        for i, (h, pi) in enumerate(zip(ij.chunk(self.nl, 1), p)):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # obj
            if n_labels:  # if any labels
                tobj[b[h], gj[h], gi[h]] = all_loss[i][3][h]
            loss[1] += self.BCEobj(pi[:, 4], tobj) * (self.balance[i] * self.hyp['obj'])

        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float()  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # # Matches
                r = t[..., 4:6] / self.anchors[i]  # wh ratio
                a = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # a = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[a]  # filter

                # # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) & a
                t = t.repeat((5, 1, 1))
                offsets = torch.zeros_like(gxy)[None] + off[:, None]
                t[..., 4:6][~j] = 0.0  # move unsuitable targets far away
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 2)  # (image, class), grid xy, grid wh
            b, c = bc.long().transpose(0, 2).contiguous()  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.transpose(0, 2).contiguous()  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 2).permute(1, 0, 2).contiguous())  # box
            tcls.append(c)  # class

            # # Unique
            # n1 = torch.cat((b.view(-1, 1), tbox[i].view(-1, 4)), 1).shape[0]
            # n2 = tbox[i].view(-1, 4).unique(dim=0).shape[0]
            # print(f'targets-unique {n1}-{n2} diff={n1-n2}')

        return tcls, tbox, indices
