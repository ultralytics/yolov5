from pathlib import Path
import tempfile
import numpy as np
import torch
import wandb
from torchvision.ops import box_iou

from utils.general import scale_coords, xywh2xyxy
from aisa_utils.dl_utils.utils import plot_object_count_difference_ridgeline


def process_batch_with_missed_labels(detections, detections_pos, labels, iouv):
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
    labels_matched = torch.zeros(labels.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections_pos[:, :4])
    matches = (iou >= iouv[0]) & (labels[:, 0:1] == detections_pos[:, 5])
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
        labels_matched = [l in matches[:, 0] for l in range(len(labels))] # iou = 0.5
    return pred_matched, labels_matched


def plot_object_count_difference_ridgeline_from_extra(extra_stats):
    ground_truths_extra = [[1] * len(e[2]) for e in extra_stats]
    preds_extra = [e[1].numpy() for e in extra_stats]
    return plot_object_count_difference_ridgeline(ground_truths_extra, preds_extra)


def log_plot_as_wandb_artifact(wand_run, fig, fig_name, temp_dir=Path(tempfile.NamedTemporaryFile().name)):
    temp_dir.mkdir(exist_ok=True)
    fig_name = f"{fig_name}.html"
    filepath = temp_dir / fig_name
    fig.write_html(open(filepath, "w"))
    artifact = wandb.Artifact("run_" + wand_run.id + fig_name, type='result')
    # Add a file to the artifact's contents
    artifact.add_file(filepath)
    # Save the artifact version to W&B and mark it as the output of this run
    wand_run.log_artifact(artifact)


def error_count_from_extra(extra_stats):
    extra_metrics = [0, 0]
    for si, extra_stat in enumerate(extra_stats):
        labels = extra_stat[2]
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
        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img1_shape, tbox, img0_shape, ratio_pad)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            preds_matched, labels_matched = process_batch_with_missed_labels(predn, predn[predn[:, 4] > threshold], labelsn, iouv)
        else:
            preds_matched, labels_matched = torch.zeros(predn.shape[0], niou, dtype=torch.bool), torch.zeros(labelsn.shape[0], dtype=torch.bool)

        labelsn = labelsn[:, 1:]
        # Get pos, negn matched and non matched to compute and show FP/FN/TP/TN
        preds_matched = preds_matched[:, 0] # iou = 0.5
        # preds_matched = np.logical_and(preds_matched, predn[:, 4] > threshold)

        predn_all.append(predn)
        preds_matched_all.append(preds_matched)
        labelsn_all.append(labelsn)
        labels_matched_all.append(labels_matched)
    images_paths = [e[0] for e in extra_stats]

    return predn_all, preds_matched_all, labelsn_all, labels_matched_all, images_paths


