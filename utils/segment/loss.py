# Ultralytics YOLOv5 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..general import xywh2xyxy
from ..loss import FocalLoss, smooth_BCE
from ..metrics import bbox_iou
from ..torch_utils import de_parallel
from .general import crop_mask


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, overlap=False):
        """
        Initializes the ComputeLoss class for YOLOv5 models with options for autobalancing and overlap handling.

        Args:
            model (torch.nn.Module): The YOLOv5 model whose loss is being computed.
            autobalance (bool, optional): Flag indicating whether to apply autobalancing of losses between output layers
                                          based on stride size (default is False).
            overlap (bool, optional): Flag indicating whether to consider overlapping bounding boxes in the loss computation
                                      (default is False).

        Attributes:
            sort_obj_iou (bool): Whether to sort objects by Intersection over Union (IoU).
            overlap (bool): Indicates if overlap handling is enabled.
            cp (torch.Tensor): Class-positive label smoothing targets.
            cn (torch.Tensor): Class-negative label smoothing targets.
            balance (list[float]): Balancing factors for different model strides.
            ssi (int): Stride 16 index if autobalance is enabled.
            BCEcls (nn.Module): Binary Cross-Entropy loss module for class predictions.
            BCEobj (nn.Module): Binary Cross-Entropy loss module for objectness predictions.
            gr (float): Gradient scaling factor.
            hyp (dict): Hyperparameters used for loss computation.
            autobalance (bool): Indicates if autobalancing is enabled.
            na (int): Number of anchors in the model.
            nc (int): Number of classes in the model.
            nl (int): Number of detection layers in the model.
            nm (int): Number of mask outputs in the model.
            anchors (torch.Tensor): Anchor boxes used in the model.

        Notes:
            - Class label smoothing is applied to improve generalization.
            - Focal loss is used to address class imbalance.

        Example:
            ```python
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            compute_loss = ComputeLoss(model, autobalance=True, overlap=True)
            ```
        """
        self.sort_obj_iou = False
        self.overlap = overlap
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
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
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.nm = m.nm  # number of masks
        self.anchors = m.anchors
        self.device = device

    def __call__(self, preds, targets, masks):  # predictions, targets, model
        """
        Computes the total YOLOv5 model loss based on predictions, targets, and masks.

        Args:
            preds (tuple[Tensor, Tensor]): A tuple containing the prediction tensors from the model.
                - `p` (Tensor): Layer-wise predictions which include bounding boxes, objectness scores, and classifications.
                - `proto` (Tensor): Proto mask used for segmentation.
            targets (Tensor): Ground truth targets containing bounding boxes and class labels.
            masks (Tensor): Ground truth masks for segmentation, used for the mask loss.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, list]: A tuple containing:
                - `lbox` (Tensor): Computed loss for bounding box regression.
                - `lobj` (Tensor): Computed loss for objectness.
                - `lcls` (Tensor): Computed loss for classification.
                - `lseg` (Tensor): Computed loss for segmentation masks.
                - `losses` (list): List containing individual loss components as (lbox, lobj, lcls, lseg).
        """
        p, proto = preds
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, self.nc, nm), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
                for bi in b.unique():
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lseg *= self.hyp["box"] / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """
        Calculates and normalizes single mask loss for YOLOv5 between predicted and ground truth masks.

        Args:
            gt_mask (torch.Tensor): Ground truth mask tensor of shape (n, H, W).
            pred (torch.Tensor): Predicted mask tensor of shape (n, nm), where nm is the number of masks.
            proto (torch.Tensor): Prototype masks tensor of shape (nm, H, W).
            xyxy (torch.Tensor): Tensor of bounding box coordinates with shape (n, 4).
            area (torch.Tensor): Tensor of mask areas with shape (n,).

        Returns:
            torch.Tensor: Calculated mask loss (scalar) normalized by the area of ground truth mask regions.

        Notes:
            This method multiplies `pred` with `proto` to generate the predicted mask and computes the binary cross-entropy
            loss with the `gt_mask`. The loss is then normalized by the corresponding mask area. This operation is performed
            for each instance in the batch, allowing backpropagation to optimize mask predictions effectively.

        Examples:
        ```python
        # Example usage
        gt_mask = torch.randn(10, 80, 80)  # Ground truth mask for 10 instances
        pred = torch.randn(10, 32)         # Predicted mask logits for 10 instances
        proto = torch.randn(32, 80, 80)    # Prototype masks
        xyxy = torch.tensor([[0, 0, 20, 20]] * 10)  # Bounding box coordinates for 10 instances
        area = torch.tensor([400.0] * 10)  # Area for each of the 10 instances

        loss_fn = ComputeLoss(model)
        loss = loss_fn.single_mask_loss(gt_mask, pred, proto, xyxy, area)
        print(loss)
        ```
        """
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

    def build_targets(self, p, targets):
        """
        Builds targets for YOLOv5 loss computation by processing input targets to match prediction shapes and anchors.

        Args:
            p (list[torch.Tensor]): List of torch tensors representing predicted feature maps from the model. Each tensor
                corresponds to a specific feature scale.
            targets (torch.Tensor): Ground truth targets with shape (num_targets, 6), where each row represents [image_idx,
                class, x, y, w, h] normalized between 0-1.

        Returns:
            tuple: A tuple containing the following elements:
                - tcls (list[torch.Tensor]): List of tensors representing target class indices for each feature map layer.
                - tbox (list[torch.Tensor]): List of tensors representing target bounding boxes for each feature map layer.
                - indices (list[tuple]): List of tuples indexing selected anchors in each feature map layer.
                - anch (list[torch.Tensor]): List of anchor tensors corresponding to selected indices for each layer.
                - tidxs (list[torch.Tensor]): List of target indices for selecting masks.
                - xywhn (list[torch.Tensor]): List of normalized target box positions [x, y, w, h] for each layer.

        Notes:
            This function plays a critical role in aligning targets with predicted outputs, facilitating the computation of
            losses for object detection and segmentation. The function considers multiple anchors and scales, applying
            geometric transformations and ensuring that targets are matched with appropriate prediction layers and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
        gain = torch.ones(8, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        if self.overlap:
            batch = p[0].shape[0]
            ti = []
            for i in range(batch):
                num = (targets[:, 0] == i).sum()  # find number of targets of each image
                ti.append(torch.arange(num, device=self.device).float().view(1, num).repeat(na, 1) + 1)  # (na, num)
            ti = torch.cat(ti, 1)  # (na, nt)
        else:
            ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na, 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[..., None]), 2)  # append anchor indices

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
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
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
            bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            (a, tidx), (b, c) = at.long().T, bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tidxs.append(tidx)
            xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])  # xywh normalized

        return tcls, tbox, indices, anch, tidxs, xywhn
