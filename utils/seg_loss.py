# TODO: merge with loss.py.. Optimize speed

import torch
from utils.torch_utils import de_parallel, is_parallel
from utils.general import xywh2xyxy, Profile
from utils.segment import mask_iou, masks_iou, crop
import torch.nn.functional as F
import torch.nn as nn
from utils.loss import smooth_BCE, FocalLoss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        self.mask_loss = MaskIOULoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(
            eps=h.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            1.0,
            h,
            autobalance,
        )
        for k in "na", "nc", "nl", "anchors", "nm":
            if hasattr(det, k):
                setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, masks=None):  # predictions, targets, model
        if masks is not None:
            return self.loss_segment(p, targets, masks)
        return self.loss_detection(p, targets)

    def loss_detection(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (
                        b[sort_id],
                        a[sort_id],
                        gj[sort_id],
                        gi[sort_id],
                        score_iou[sort_id],
                    )
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
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def loss_segment(self, preds, targets, masks):
        """
        proto_out:[batch-size, mask_dim, mask_hegiht, mask_width]
        masks:[batch-size * num_objs, image_height, image_width]
        每张图片objects数量不同，到时候处理时填充不足的
        """
        p = preds[0]
        proto_out = preds[1]
        mask_h, mask_w = proto_out.shape[2:]
        proto_out = proto_out.permute(0, 2, 3, 1)

        device = targets.device
        lcls, lbox, lobj, lseg = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        tcls, tbox, indices, anchors, tidxs, xywh = self.build_targets_for_masks(
            p, targets
        )  # targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (
                        b[sort_id],
                        a[sort_id],
                        gj[sort_id],
                        gi[sort_id],
                        score_iou[sort_id],
                    )
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, self.nm :], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, self.nm :], t)  # BCE

                # Mask Regression
                mask_gt = masks[tidxs[i]]
                downsampled_masks = F.interpolate(
                    mask_gt[None, :],
                    (mask_h, mask_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                mxywh = xywh[i]
                mws, mhs = mxywh[:, 2:].T
                mws, mhs = mws / pi.shape[3], mhs / pi.shape[2]
                mxywhs = (
                    mxywh
                    / torch.tensor(pi.shape, device=mxywh.device)[[3, 2, 3, 2]]
                    * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=mxywh.device)
                )
                mxyxys = xywh2xyxy(mxywhs)

                batch_lseg = torch.zeros(1, device=device)
                for bi in b.unique():
                    index = b == bi
                    mask_gti = downsampled_masks[index]
                    mask_gti = mask_gti.permute(1, 2, 0).contiguous()

                    mw, mh = mws[index], mhs[index]
                    mxyxy = mxyxys[index]
                    psi = ps[index][:, 5 : self.nm]
                    proto = proto_out[bi]

                    batch_lseg += self.single_mask_loss(mask_gti, psi, proto, mxyxy, mw, mh)
                lseg += batch_lseg / len(b.unique())

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lseg *= self.hyp["box"]
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, w, h):
        """mask loss of single pic."""
        # (80, 80, 32) @ (32, n) -> (80, 80, n)
        pred_mask = proto @ pred.tanh().T
        lseg = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        lseg = crop(lseg, xyxy)
        lseg = lseg.mean(dim=(0, 1)) / w / h
        return lseg.mean()

    def mask_loss(self, gt_masks, preds, protos, xyxys, ws, hs):
        """mask loss of batches."""
        pass

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
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
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def build_targets_for_masks(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tidxs, xywh = [], [], [], [], [], []
        gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        ti = (
            torch.arange(nt, device=targets.device).float().view(1, nt).repeat(na, 1)
        )  # same as .repeat_interleave(nt)

        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None], ti[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
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
            tidx = t[:, 7].long()
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tidxs.append(tidx)
            xywh.append(torch.cat((gxy, gwh), 1))

        return tcls, tbox, indices, anch, tidxs, xywh
    

class MaskIOULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_mask, gt_mask, mxyxy=None):
        """
        Args:
            pred_mask (torch.Tensor): prediction of masks, (80/160, 80/160, n)
            gt_mask (torch.Tensor): ground truth of masks, (80/160, 80/160, n)
            mxyxy (torch.Tensor): ground truth of boxes, (n, 4)
        """
        _, _, n = pred_mask.shape  # same as gt_mask
        pred_mask = pred_mask.sigmoid()
        if mxyxy is not None:
            pred_mask = crop(pred_mask, mxyxy)
            gt_mask = crop(gt_mask, mxyxy)
        pred_mask = pred_mask.permute(2, 0, 1).view(n, -1)
        gt_mask = gt_mask.permute(2, 0, 1).view(n, -1)
        iou = masks_iou(pred_mask, gt_mask)
        return 1.0 - iou

import math 

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU