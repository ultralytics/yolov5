# Loss functions
import torch
import torch.nn as nn

from utils.general_polygon import order_corners, polygon_bbox_iou, wh_iou
from utils.loss import *
from utils.torch_utils import is_parallel


class Polygon_ComputeLoss:
    # Compute losses for polygon anchors

    def __init__(self, model, autobalance=False):
        super().__init__()
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

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Polygon_Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))


    def __call__(self, p, targets):  # predictions, targets, model
        """"
            targets: total anchors for this batch x 10
            p: nl (number of anchor layers) x bs x na (number of anchors per layer)
              x ny (grid width) x nx (grid height) x no (89, number of outputs per anchor)
            self.anchors: nl (number of prediction layers) x na (number of anchors per layer) x 2 (width and height)
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets for computing loss

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # pi: bs x na x ny x nx x no
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj, shape is bs x na x ny x nx

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pbox = ps[:, :8]  # predicted polygon box

                # tbox[i] is ordered, and pbox from model will learn the
                # sequential order naturally: so specify ordered=True
                iou = polygon_bbox_iou(pbox, tbox[i], CIoU=True, device=device, ordered=True)  # iou(prediction, target)
                # lbox += (1.0 - iou).mean() # iou loss

                zero = torch.tensor(0., device=device)
                # Include the restrictions on predicting sequence: y3, y4 >=
                # y1, y2; x1 <= x2; x4 <= x3
                lbox += (torch.max(zero, ps[:, 1] - ps[:, 5]) ** 2).mean() / 6 + (torch.max(zero, ps[:, 3] - ps[:, 5]) ** 2).mean() / 6 + \
                        (torch.max(zero, ps[:, 1] - ps[:, 7]) ** 2).mean() / 6 + (torch.max(zero, ps[:, 3] - ps[:, 7]) ** 2).mean() / 6 + \
                        (torch.max(zero, ps[:, 0] - ps[:, 2]) ** 2).mean() / 6 + (torch.max(zero, ps[:, 6] - ps[:, 4]) ** 2).mean() / 6
                # include the values of each vertice of poligon into loss
                # function
                lbox += nn.SmoothL1Loss(beta=0.11)(pbox, tbox[i])

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 9:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 9:], t)  # BCE
                    # lcls += nn.CrossEntropyLoss()(ps[:, 9:],
                                                                         # t.long().argmax(dim=1)) # softmax loss

            obji = self.BCEobj(pi[..., 8], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()


    def build_targets(self, p, targets):
        """
            Build targets for Polygon_ComputeLoss
            p: nl x bs x na x ny x nx x no
            targets: image,class,x1,y1,x2,y2,x3,y3,x4,y4 (x1, y1...represent the relative positions)
        """

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain has 11 elements: img id, class id, xyxyxyxy, anchor id
        gain = torch.ones(11, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1], # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # self.anchors have shape nl x na x 2
            anchors = self.anchors[i]
            gain[2:10] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2]]  # xyxyxyxy gain

            # Match targets to anchors
            t = targets * gain  # now t is unnormalized to pixel-level, with shape na x nt x 11
            if nt:
                # Utilize minimum bounding box to select boxes
                t_width = (t[..., 2:10:2].max(dim=-1)[0] - t[..., 2:10:2].min(dim=-1)[0])[..., None]
                t_height = (t[..., 3:10:2].max(dim=-1)[0] - t[..., 3:10:2].min(dim=-1)[0])[..., None]
                wh = torch.cat((t_width, t_height), dim=-1)

                # Using shape matches
                # r = wh / anchors[:, None] # wh ratio
                # j = torch.max(r, 1.  / r).max(2)[0] < self.hyp['anchor_t'] #
                # compare

                # Consider only best anchors
                # max_ious, max_ious_idx = wh_iou(anchors, wh[0]).max(dim=0)
                # mask = max_ious > self.hyp['iou_t']
                # t = t[max_ious_idx[mask], mask]

                # Consider all anchors that exceed the iou threshold
                j = wh_iou(anchors, wh[0]) > self.hyp['iou_t'] # iou criterion
                t = t[j]  # filter

                # now t has shape nt x 11
                # Offsets
                center_x = t[:, 2:10:2].mean(dim=-1)[:, None]
                center_y = t[:, 3:10:2].mean(dim=-1)[:, None]
                gxy = torch.cat((center_x, center_y), dim=-1)  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # Utilize polygon center to compute the grid indices gi, gj
            b, c = t[:, :2].long().T  # image, class
            center_x = t[:, 2:10:2].mean(dim=-1)[:, None]
            center_y = t[:, 3:10:2].mean(dim=-1)[:, None]
            gxy = torch.cat((center_x, center_y), dim=-1)  # grid xy
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            # tbox represents the relative positions from xyxyxyxy to center
            # (in grid)
            t[:, 2:10] = order_corners(t[:, 2:10])
            a = t[:, 10].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices

            # same corners, different center points, different relative
            # positions
            tbox.append(t[:, 2:10] - gij.repeat(1, 4))  # polygon box
            # different corners, different center points, same relative
                                                                  # positions
                                                                  # gij_origin = (gxy-0).long()
                                                                  # tbox.append(t[:, 2:10]-gij_origin.repeat(1, 4))

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
