# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """
    Returns label-smoothed Binary Cross Entropy (BCE) targets to mitigate overfitting.

    Args:
        eps (float, optional): Smoothing factor for BCE targets. A value between 0 and 1 where 0 equates to no smoothing,
            and higher values increase the amount of smoothing. Default is 0.1.

    Returns:
        tuple[float, float]: A tuple containing the positive and negative target values after applying label smoothing.
            The positive target is `1.0 - 0.5 * eps` and the negative target is `0.5 * eps`.

    Note:
        Refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441 for detailed discussion on label
        smoothing implementation and its benefits.

    Example:
        ```python
        pos, neg = smooth_BCE(0.1)
        print(pos, neg)  # 0.95 0.05
        ```
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """
        Initializes a modified BCEWithLogitsLoss with reduced missing label effects, incorporating an optional alpha
        smoothing parameter.

        Args:
            alpha (float, optional): Smoothing factor to mitigate the impact of missing labels. Default is 0.05.

        Returns:
            None

        Notes:
            This custom loss function leverages the standard `nn.BCEWithLogitsLoss` to compute the loss, but applies
            additional handling to reduce the side effects caused by missing labels.

        Examples:
            ```python
            from ultralytics import BCEBlurWithLogitsLoss

            criterion = BCEBlurWithLogitsLoss(alpha=0.1)
            loss = criterion(predictions, targets)
            ```
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """
        Computes the Binary Cross-Entropy loss with Logits for YOLOv5, reducing the effect of missing labels.

        Args:
            pred (torch.Tensor): Predicted tensor from the model, containing logits.
            true (torch.Tensor): Ground truth tensor, containing binary labels.

        Returns:
            torch.Tensor: The computed BCEBlurWithLogits loss, encapsulated in a tensor.

        Notes:
            This function applies a modified BCEWithLogitsLoss that primarily reduces the impact of missing labels
            by scaling the loss with an alpha factor derived from the predictions and ground truths. The
            adjustment uses an exponential scaling factor based on the difference between predicted and true
            values.

        Example:
            ```python
            criterion = BCEBlurWithLogitsLoss(alpha=0.05)
            pred = torch.tensor([0.2, 0.7, 0.1])
            true = torch.tensor([0, 1, 0])
            loss = criterion(pred, true)
            print(loss)  # Output: Tensor comprising the computed loss value
            ```
        """
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
        """
        Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to 'none'.

        Args:
            loss_fcn (nn.Module): The base loss function to which focal loss is applied. Must be an instance of
                nn.BCEWithLogitsLoss.
            gamma (float): The focusing parameter that adjusts the rate at which easy examples are down-weighted. Default is 1.5.
            alpha (float): The balance parameter that adjusts the importance of positive vs negative examples. Default is 0.25.

        Returns:
            None

        Examples:
            ```python
            import torch.nn as nn
            from ultrlalytics.loss import FocalLoss

            # Example of initializing FocalLoss with default parameters
            loss = nn.BCEWithLogitsLoss()
            focal_loss = FocalLoss(loss_fcn=loss)
            ```
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """
        Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss.

        Args:
            pred (torch.Tensor): Predicted tensor with logits. Shape should be (N, \*) where N is the batch size.
            true (torch.Tensor): Ground truth tensor with labels. Shape should match `pred`.

        Returns:
            torch.Tensor: Computed focal loss. Returns a scalar if `reduction` is 'mean' or 'sum', otherwise a tensor with same shape as input.

        Notes:
            The focal loss is designed to address class imbalance by down-weighting easy examples and focusing more on hard examples.
            This implementation wraps around `nn.BCEWithLogitsLoss` and modifies the loss with alpha and gamma factors.

            For more details, see:
            https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py

        Examples:
            ```
            criterion = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5, alpha=0.25)
            loss = criterion(pred, true)
            ```
        """
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
        """
        Initializes Quality Focal Loss with a specified loss function, gamma, and alpha values; modifies loss reduction
        to 'none'.

        Args:
            loss_fcn (nn.Module): The loss function to be wrapped, typically `nn.BCEWithLogitsLoss()`.
            gamma (float): Focusing parameter that controls the rate at which easy examples are down-weighted (default is 1.5).
            alpha (float): Balancing parameter to balance the importance of positive/negative examples (default is 0.25).

        Returns:
            None

        Notes:
            The `reduction` attribute of the provided loss function is changed to 'none' to apply Quality Focal Loss to each element individually. Ensure `loss_fcn` is an instance of `nn.BCEWithLogitsLoss`.

        Example:
            ```python
            import torch.nn as nn
            from your_module import QFocalLoss

            criterion = QFocalLoss(nn.BCEWithLogitsLoss(), gamma=2.0, alpha=0.5)
            ```
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """
        Computes the Quality Focal Loss between predicted and true labels using BCEWithLogitsLoss, adjusting for class
        imbalance with gamma and alpha.

        Args:
            pred (torch.Tensor): Predicted logits from the model, with shape (N, *) where N is the batch size.
            true (torch.Tensor): Ground truth labels, with shape (N, *) where N is the batch size. Should be binary (0 or 1).

        Returns:
            torch.Tensor: Computed Quality Focal Loss. If `reduction` is 'mean', returns a scalar tensor; if 'sum', returns
            a summed scalar tensor; if 'none', returns a tensor of the same shape as input predictions.

        Note:
            The `reduction` attribute of the provided loss function is modified within this method to ensure correct
            application of focal loss.

        Examples:
            ```python
            loss_fcn = nn.BCEWithLogitsLoss(reduction='mean')
            q_focal_loss = QFocalLoss(loss_fcn, gamma=1.5, alpha=0.25)
            pred = torch.randn(8, 10, requires_grad=True)  # Example batch of 8 samples with 10 logits each
            true = torch.randint(0, 2, (8, 10), dtype=torch.float32)  # Example batch of 8 samples with 10 binary labels each
            loss = q_focal_loss(pred, true)
            loss.backward()
            ```
        """
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


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """
        Initializes the ComputeLoss class, setting up components based on the model and specified hyperparameters.

        Args:
            model (torch.nn.Module): The model to compute loss for, typically a YOLOv5 model.
            autobalance (bool): If True, enables automatic balancing of losses across different scales.

        Returns:
            None

        Notes:
            - The function initializes several loss components, including BCE with logits loss for classification and
              objectness scores. It also sets up focal loss if specified by the model's hyperparameters.
            - The `smooth_BCE` function is used to apply label smoothing to the BCE loss.
            - Automatically determines device and model properties like the number of layers (nl), number of classes (nc),
              number of anchors (na), and anchor configurations.
            - Sets up a mechanism for balancing losses across layers, with preset balance values for different layer depths.

        Example:
            ```python
            model = torch.load('yolov5s.pt')
            compute_loss = ComputeLoss(model, autobalance=True)
            ```
        """
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
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """
        Calculates class, box, and object loss for given predictions and targets in a forward pass.

        Args:
            p (list[Tensor]): List of tensors containing the model's layer predictions. Each tensor is expected to be
                of shape (batch_size, num_anchors, grid_height, grid_width, num_predictions_per_grid_cell).
            targets (Tensor): Tensor containing the taget annotations. Expected to be of shape (num_targets, 6) where each
                row represents (image_index, class_label, x_center, y_center, width, height).

        Returns:
            tuple[Tensor, Tensor, Tensor]: The calculated losses for classification, bounding box, and objectness respectively.

        Notes:
            The function is implemented to work with YOLOv5 models and calculates three types of losses:
            - Class loss using BCEWithLogitsLoss
            - Box loss using IoU
            - Objectness loss using BCEWithLogitsLoss

            Class label smoothing and focal loss are applied if specified in the model's hyperparameters.

            The final loss values are scaled by their respective weights from the model's hyperparameters and adjusted
            for auto-balancing if enabled.
        """
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
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

    def build_targets(self, p, targets):
        """
        Prepares model targets from input targets for loss computation, returning class, box, indices, and anchors.

        Args:
            p (torch.Tensor): Predictions from the model, typically in the shape [batch, anchors, grid, grid, predict_params].
            targets (torch.Tensor): Ground truth targets in the shape [num_targets, 6], with each target being (image_index,
                                    class, x, y, w, h).

        Returns:
            tuple: A tuple containing:
                - tcls (list of torch.Tensor): List of class targets for each prediction layer.
                - tbox (list of torch.Tensor): List of box targets for each prediction layer.
                - indices (list of tuples): List of indices for each prediction layer. Each tuple contains (batch, anchor, y, x).
                - anch (list of torch.Tensor): List of anchor box tensors for each prediction layer.

        Notes:
            - The function handles anchor matching to targets based on IoU and adjusts targets to be suitable for computing
              the loss with the predictions.
            - Offsets are applied to handle objects near grid boundaries enabling better localization accuracy.

        Examples:
            >>> compute_loss = ComputeLoss(model)
            >>> tcls, tbox, indices, anch = compute_loss.build_targets(predictions, targets)
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

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
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
