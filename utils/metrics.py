# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    """
    Calculates the fitness of a model based on the weighted sum of key performance metrics: Precision (P), Recall (R),
    mean Average Precision at IoU 0.5 (mAP@0.5), and mean Average Precision at IoU 0.5:0.95 (mAP@0.5:0.95).

    Args:
        x (np.ndarray): Array of input metrics in the form [P, R, mAP@0.5, mAP@0.5:0.95].

    Returns:
        float: Weighted sum of the input metrics, representing the model fitness.

    Notes:
        The specific weights used are [0.0, 0.0, 0.1, 0.9], which means the fitness score is highly influenced by
        the mAP@0.5:0.95 metric.

    Example:
        ```python
        metrics = np.array([0.9, 0.85, 0.75, 0.8])
        fit_score = fitness(metrics)
        print(f"Model Fitness: {fit_score}")
        ```
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """
    Applies box filter smoothing to an input array `y` using a specified smoothing fraction `f`.

    Args:
        y (np.ndarray): Input array to be smoothed.
        f (float, optional): Fraction of the array length used to define the size of the smoothing
            kernel. Default is 0.05.

    Returns:
        np.ndarray: Smoothed array of the same shape as input `y`.

    Notes:
        The function uses a simple box filter for smoothing. The number of filter elements is computed
        based on the length of the input array and the fraction `f`, ensuring it is always odd for
        symmetric padding. Padding is applied to the start and end of the array to mitigate boundary
        effects.

    Examples:
        ```python
        import numpy as np
        from ultralytics import smooth

        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_smooth = smooth(y, f=0.1)
        print(y_smooth)
        ```
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision (AP) for each class given true positives, confidence scores, predicted and target
    classes.

    Args:
        tp (np.ndarray): True positives, with shape (N, 1) or (N, 10), where N is the number of predictions.
        conf (np.ndarray): Confidence scores for the predictions, values ranging from 0 to 1.
        pred_cls (np.ndarray): Predicted classes for each prediction.
        target_cls (np.ndarray): True classes corresponding to each prediction.
        plot (bool, optional): If True, plots the precision-recall curve at mAP@0.5. Defaults to False.
        save_dir (str, optional): Directory to save the plotted curves. Defaults to ".".
        names (tuple, optional): Tuple of class names.
        eps (float, optional): Small epsilon value to prevent division by zero. Defaults to 1e-16.
        prefix (str, optional): Prefix for saved plot filenames. Defaults to "".

    Returns:
        tuple: Containing:
            - np.ndarray: Average precision (AP) for each class, shape (num_classes, 10).
            - np.ndarray: Precision for each class, interpolated at 1000 points.
            - np.ndarray: Recall for each class, interpolated at 1000 points.
            - np.ndarray: True positive count per class.
            - np.ndarray: False positive count per class.
            - np.ndarray: Unique target classes.

    Notes:
        - This function computes the precision-recall curves for each class and interpolates them to 1000 points for
          plotting.
        - Precision (P) and recall (R) metrics, along with their harmonic mean (F1 score), are used for mAP calculations.

    Example:
        ```python
        tp = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        conf = np.array([0.9, 0.6, 0.3])
        pred_cls = np.array([0, 1, 2])
        target_cls = np.array([0, 1, 2])
        plot = True
        save_dir = "./results"

        ap, p, r, tp, fp, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls, plot, save_dir)
        ```
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given a recall and precision curve.

    Args:
        recall (list[float]): Recall values.
        precision (list[float]): Precision values.

    Returns:
        tuple[float, ndarray, ndarray]: Returns a tuple containing:
            - Average precision (float): The area under the precision-recall curve.
            - Precision curve (ndarray): The precision values used to compute AP.
            - Recall curve (ndarray): The recall values used to compute AP.

    Notes:
        The precision-recall curve is computed by interpolating the precision values at different recall thresholds.
        This function uses 101-point interpolation as used in COCO evaluation.

    Example:
        ```python
        recall = [0.1, 0.4, 0.7, 0.9]
        precision = [1.0, 0.8, 0.6, 0.4]
        ap, precision_curve, recall_curve = compute_ap(recall, precision)
        ```
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        Initializes a confusion matrix for evaluating object detection models.

        Args:
            nc (int): Number of classes to consider in the confusion matrix.
            conf (float, optional): Confidence threshold for considering detections. Default is 0.25.
            iou_thres (float, optional): Intersection over Union (IoU) threshold for considering a detection as
                true positive. Default is 0.45.

        Returns:
            None

        Notes:
            The confusion matrix is initially filled with zeros and has dimensions (nc + 1, nc + 1).
            The extra row/column accounts for the background or no-object class.

        Examples:
            ```python
            # Initialize a confusion matrix for a model with 3 classes, confidence threshold of 0.3,
            # and IoU threshold of 0.5
            cm = ConfusionMatrix(nc=3, conf=0.3, iou_thres=0.5)
            ```
        """
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Processes a batch of detections and labels, updating the confusion matrix.

        Args:
            detections (np.ndarray): Array of shape (N, 6) where each row contains [x1, y1, x2, y2, conf, class]. Represents the detected bounding boxes with their coordinates, confidence scores, and predicted classes.
            labels (np.ndarray): Array of shape (M, 5) where each row contains [class, x1, y1, x2, y2]. Represents the ground truth bounding boxes with their classes and coordinates.

        Returns:
            None

        Notes:
            This function modifies the confusion matrix attribute of the class instance. It uses an IoU threshold to determine matches between detected and ground truth boxes, updating the matrix with true positives, false positives, and false negatives accordingly. The detection boxes must have a confidence score above a certain threshold to be considered.

            Example:

            ```python
            cm = ConfusionMatrix(nc=5)  # Initialize for 5 classes
            detections = np.array([[50, 50, 150, 150, 0.9, 1], [30, 30, 100, 100, 0.75, 2]])
            labels = np.array([[1, 50, 50, 150, 150], [2, 30, 30, 100, 100]])
            cm.process_batch(detections, labels)
            ```
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        """
        Calculates true positives (tp) and false positives (fp) excluding the background class from the confusion
        matrix.

        Returns:
            tp (np.ndarray): Array of true positives for each class excluding the background.
            fp (np.ndarray): Array of false positives for each class excluding the background.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """
        Plots a confusion matrix using seaborn with optional normalization and saves it to a specified directory.

        Args:
            normalize (bool): If True, normalizes the confusion matrix by column sums. Defaults to True.
            save_dir (str | Path): Directory where the plot will be saved. Defaults to an empty string.
            names (list): List of class names corresponding to the matrix indices. Defaults to an empty tuple.

        Returns:
            None: The function saves the confusion matrix plot to the specified directory and does not return any value.

        Notes:
            - This function uses seaborn for visualization.
            - The confusion matrix is normalized by column sums if `normalize` is set to True. This means each value in the matrix
              is divided by the sum of its column. Elements below a threshold of 0.005 are not annotated in the heatmap.
            - The function saves the plot as 'confusion_matrix.png' in the directory specified by `save_dir`.

        Example:
            ```python
            cm = ConfusionMatrix(nc=10)  # Initialize with 10 classes
            cm.plot(normalize=True, save_dir="results/", names=["class1", "class2", ..., "class10"])
            ```

        Raises:
            Warning: If plotting fails, a warning message is issued.
        """
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self):
        """
        Prints the confusion matrix row-wise, with each class and its predictions separated by spaces.

        Args:
            None

        Returns:
            None

        Examples:
            ```python
            # Assume cm is an initialized instance of ConfusionMatrix
            cm.print()
            ```
            This will print the confusion matrix to the standard output, with each row corresponding to a class or background
            and each column representing the predicted class counts.
        """
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU), Generalized IoU (GIoU), Distance IoU (DIoU), or Complete IoU (CIoU)
    between two bounding boxes.

    Args:
      box1 (tensor): First bounding box, of shape (1, 4), in either xywh or xyxy format.
      box2 (tensor): Second bounding box, of shape (n, 4), in either xywh or xyxy format.
      xywh (bool): If True, bounding boxes are in (x_center, y_center, width, height) format, else (x1, y1, x2, y2).
      GIoU (bool): If True, compute Generalized IoU. Default is False.
      DIoU (bool): If True, compute Distance IoU. Default is False.
      CIoU (bool): If True, compute Complete IoU. Default is False.
      eps (float): Small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
      tensor: Calculated IoU, GIoU, DIoU, or CIoU values, depending on the selected computation type.

    Notes:
      IoU, or Intersection over Union, measures the overlap between two bounding boxes. GIoU, DIoU, and CIoU are
      extensions of IoU that include additional factors to optimize object detection models more effectively.
      For more details, refer to the respective publications:
        - GIoU: https://arxiv.org/pdf/1902.09630.pdf
        - DIoU and CIoU: https://arxiv.org/abs/1911.08287v1
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Returns:
        Tensor: Intersection-over-Union (IoU) matrix with shape (N, M), where each entry corresponds to the IoU value
        between a box in `box1` and a box in `box2`.

    Examples:
        ```python
        import torch
        from ultralytics.utils.metrics import box_iou

        box1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]])
        box2 = torch.tensor([[1.0, 1.0, 4.0, 4.0]])

        iou_matrix = box_iou(box1, box2)
        print(iou_matrix)
        # Output: tensor([[0.1429, 0.0000]])
        ```
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Returns the intersection over box2's area given box1 and box2 coordinates.

    Args:
        box1 (np.ndarray): A 1D array of shape (4,) representing a single bounding box in the format (x1, y1, x2, y2).
        box2 (np.ndarray): A 2D array of shape (n, 4) representing n bounding boxes in the format (x1, y1, x2, y2).
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        np.ndarray: A 1D array of shape (n,) containing the intersection over box2 area for each box2 with box1.

    Example:
        ```python
        box1 = np.array([0, 0, 2, 2])
        box2 = np.array([[1, 1, 3, 3], [0, 0, 1, 1]])
        ioa = bbox_ioa(box1, box2)
        print(ioa)  # Output: array([0.25, 1.0])
        ```

    Notes:
        - The function assumes that the box coordinates are in the format (x1, y1, x2, y2).
        - The intersection area computation clips the result to ensure non-negative values.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU) for two sets of widths and heights.

    Args:
        wh1 (torch.Tensor): Tensor of shape (N, 2) representing widths and heights.
        wh2 (torch.Tensor): Tensor of shape (M, 2) representing widths and heights.
        eps (float): Small epsilon value to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: IoU values as a tensor of shape (N, M).

    Notes:
        - Both `wh1` and `wh2` should have two columns where the first column represents widths and the second represents heights.
        - The function computes pairwise IoU between each width-height pair in `wh1` and each width-height pair in `wh2`.

    Examples:
        ```python
        import torch
        from ultralytics import wh_iou

        wh1 = torch.tensor([[2, 3], [3, 4]])
        wh2 = torch.tensor([[1, 2], [2, 3], [3, 5]])

        iou = wh_iou(wh1, wh2)
        print(iou)
        ```
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """
    Plots precision-recall curves, optionally per class, and saves the plot to a specified directory.

    Args:
        px (list[float]): List of recall values for plotting the x-axis.
        py (list[list[float]]): List of lists containing precision values for each class, used for plotting the y-axis.
        ap (np.ndarray): Array of average precision values of shape (N, 2), where N is the number of classes.
        save_dir (Path | str, optional): Directory to save the resulting plot. Defaults to Path("pr_curve.png").
        names (list[str], optional): List of class names corresponding to the classes in the dataset. Defaults to ().

    Returns:
        None

    Examples:
        ```python
        px = np.linspace(0, 1, 1000)
        py = [[0.9, 0.8, 0.7, ..., 0.0], [0.85, 0.75, 0.65, ..., 0.0], ...]
        ap = np.array([[0.85, 0], [0.75, 0], ...])  # Each row containing [ap, 0]
        names = ['class1', 'class2', ...]
        plot_pr_curve(px, py, ap, save_dir=Path("results/pr_curve.png"), names=names)
        ```

    Notes:
        - This function is intended to be used as part of a multi-threaded plot generation.
        - The function will display a per-class legend if the number of classes is fewer than 21.
        - Precision-recall curves are visual aids for evaluating the performance of object detection models.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """
    Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing.

    Args:
        px (np.ndarray): Array of confidence threshold values.
        py (np.ndarray): Array of metric values corresponding to each threshold in `px`.
        save_dir (Path | str): Directory to save the plot image.
        names (tuple): Names of the classes, used for the plot legend.
        xlabel (str): Label for the x-axis. Defaults to 'Confidence'.
        ylabel (str): Label for the y-axis, representing the performance metric. Defaults to 'Metric'.

    Returns:
        None

    Note:
        This function utilizes smoothing to provide a more visually appealing curve.

    Example:
        ```python
        px = np.linspace(0, 1, 100)
        py = np.array([...])  # Shape: (num_classes, 100)
        plot_mc_curve(px, py, save_dir='metric_conf_curve.png', names=('class1', 'class2'), xlabel='Confidence', ylabel='Precision')
        ```

    Note:
        The resulting plot will be saved in the specified `save_dir` location.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
