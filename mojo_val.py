import numpy as np
import torch
from torchvision.ops import box_iou

from utils.general import scale_coords, xywh2xyxy
from aisa_utils.dl_utils.utils import plot_object_count_difference_ridgeline


def process_batch_with_missed_labels(detections, labels, iouv):
    """
    Return matched predictions' and labels' matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        true_positive (Array[N, 10]), for 10 IoU levels
        correct (Array[M, 10]), for 10 IoU levels
    """
    pred_matched = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    targets_matched = torch.zeros(labels.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    matches = (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
    x = torch.where(matches)  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                            1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]  # Sort by IoU
            matches = matches[
                np.unique(matches[:, 1], return_index=True)[1]]  # Remove one prediction matching two targets
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[
                np.unique(matches[:, 0], return_index=True)[1]]  # Remove one label matching two predictions
        matches = torch.Tensor(matches).to(iouv.device)
        matches_by_iou = matches[:, 2:3] >= iouv
        pred_matched[matches[:, 1].long()] = matches_by_iou
        targets_matched = [l in matches_by_iou[:, 0] for l in range(len(labels))] # iou = 0.5
    return pred_matched, targets_matched


def plot_object_count_difference_ridgeline_from_extra(extra_stats):
    ground_truths_extra = [[1] * len(e[2]) for e in extra_stats]
    preds_extra = [e[1].numpy() for e in extra_stats]
    return plot_object_count_difference_ridgeline(ground_truths_extra, preds_extra)


def error_count_from_extra(extra_stats):
    extra_metrics = [0, 0]
    for si, extra_stat in enumerate(extra_stats):
        targets = extra_stat[2]
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        pred_conf = extra_stat[1][:, 4].numpy()
        extra_metrics[0] += np.abs(nl - len(np.where(pred_conf >= 0.3)[0]))
        extra_metrics[1] += np.abs(nl - len(np.where(pred_conf >= 0.5)[0]))
        extra_metrics = [v / len(extra_stats) for v in extra_metrics]
    return extra_metrics


def compute_predictions_and_labels(extra_stats, *, threshold):
    """
    Compute predictions and labels based on extra_stats given by the ultralytics val.py main loop
    Arguments:
        extra_stats Array[size of val dataset, 6]
            List of path_to_image, predictions, labels, shape_in, shape_out, padding_ratio
    Returns:

    """
    iouv = torch.linspace(0.5, 0.95, 10)
    single_cls = True
    niou = len(iouv)

    predn_all, preds_matched_all, labelsn_all, labels_matched_all = [], [], [], []

    for image_idx, (path, pred, labels, img1_shape, img0_shape, ratio_pad) in enumerate(extra_stats):
        nl = len(labels)

        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_coords(img1_shape, predn[:, :4], img0_shape, ratio_pad)  # native-space pred
        predn_positive = predn[predn[:, 4] >= threshold]
        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img1_shape, tbox, img0_shape, ratio_pad)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            preds_matched, labels_matched = process_batch_with_missed_labels(predn_positive, labelsn, iouv)
        else:
            preds_matched = torch.zeros(predn_positive.shape[0], niou, dtype=torch.bool)
            labels_matched = torch.zeros(labelsn.shape[0], dtype=torch.bool)

        labelsn = labelsn[:, 1:]
        # Get pos, negn matched and non matched to compute and show FP/FN/TP/TN
        preds_matched = preds_matched[:, 0] # iou = 0.5

        predn_all.append(predn)
        preds_matched_all.append(preds_matched)
        labelsn_all.append(labelsn)
        labels_matched_all.append(labels_matched)
    images_paths = [e[0] for e in extra_stats]

    return predn_all, preds_matched_all, labelsn_all, labels_matched_all, images_paths


