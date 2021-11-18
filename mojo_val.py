import numpy as np
import torch
import cv2
from aisa_utils.dl_utils.utils import plot_object_count_difference_ridgeline
from torchvision.ops import box_iou

from plotly.subplots import make_subplots
import plotly.express as px

from utils.general import save_one_box
from utils.general import scale_coords, xywh2xyxy, clip_coords, xyxy2xywh


def process_batch_with_missed_labels(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
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


def draw_targets(frame, bboxes, matches, mask, rect=False):
    assert bboxes.shape[0] == matches.shape[0] == mask.shape[0]
    clip_coords(bboxes, frame.shape)
    bboxes.astype(int)
    if not rect:
        bboxes = xyxy2xywh(bboxes)
    for match, bbox, m in zip(matches, bboxes, mask):
        if m:
            bbox = [int(_) for _ in bbox]
            if match:
                color = (0, 255, 0)
                radius = 4
            else:
                color = (0, 0, 255)
                radius = 2
            if not rect:
                frame = cv2.circle(frame, bbox[:2], radius, color, radius)
            else:
                frame = cv2.rectangle(frame, bbox[:2], bbox[2:], color, 4)


def generate_crop(extra_stats, predn_with_path):
    crop_box = predn_with_path[:4]
    frame_idx = int(predn_with_path[-1])  # paths idx should be at the end
    frame_path = extra_stats[frame_idx][0]
    frame = cv2.imread(str(frame_path), 1)

    frame_crop = save_one_box(crop_box, frame, gain=1.10, BGR=False, save=False)
    return frame_crop


def plot_crops(crops_low, crops_high, title, gt_pos):
    titles_low = [f"{'💔' if gt_pos else '💚' } Low conf (idx: {idx})" for idx in range(len(crops_low))]
    titles_high = [f"{'💚' if gt_pos else '💔' } High conf (idx: {idx})" for idx in range(len(crops_high))]
    crops = crops_low + crops_high
    titles = titles_low + titles_high
    fig = make_subplots(
        rows=len(crops) // 5 + 1,
        cols=5,
        subplot_titles=titles
    )

    for n, image in enumerate(crops):
        fig.add_trace(px.imshow(image).data[0], row=int(n / 5) + 1, col=n % 5 + 1)

    # the layout gets lost, so we have to carry it over - but we cannot simply do
    # fig.layout = layout since the layout has to be slightly different for subplots
    # fig.layout.yaxis in a subplot refers only to the first axis for example
    # update_yaxes updates *all* axis on the other hand
    if len(crops):
        layout = px.imshow(crops[0], color_continuous_scale='gray').layout
        fig.layout.coloraxis = layout.coloraxis
        fig.update_xaxes(**layout.xaxis.to_plotly_json())
        fig.update_yaxes(**layout.yaxis.to_plotly_json())
    fig.layout.title = title
    return fig


def plot_fp_fn(extra_stats):
    iouv = torch.linspace(0.5, 0.95, 10)
    single_cls = True
    niou = len(iouv)
    true_pos, true_neg, false_pos, false_neg = [], [], [], []
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
            preds_matched, targets_matched = process_batch_with_missed_labels(predn, labelsn, iouv)
        else:
            preds_matched, targets_matched = torch.zeros(pred.shape[0], niou, dtype=torch.bool), torch.zeros(labelsn.shape[0], dtype=torch.bool)

        labelsn = labelsn[:, 1:]
        # Confidence threshold for FP/FN/TP/TN
        conf_thresh = 0.1
        preds_matched = preds_matched[:, 0] # iou = 0.5
        preds_not_matched = np.logical_not(preds_matched)
        targets_not_matched = np.logical_not(targets_matched)
        preds_pos = predn[:, 4] >= conf_thresh
        preds_neg = predn[:, 4] < conf_thresh

        # Draw one image
        if False:

            # Draw targets that are missed or not
            draw_targets(frame, labelsn.numpy(), np.logical_not(labels_miss.numpy()), np.ones(labels.shape[0]),  rect=True)
            # Draw wrong and good preds that are above a certain threshold
            draw_targets(frame, predn[:, :4].numpy(), predn[preds_matched, 0].numpy(), predn[:, 4] > conf_thresh)

            cv2.imshow("Draw image with targets and preds", frame)
            cv2.waitKey()
            cv2.destroyAllWindows()

        predn_with_path = torch.cat([predn, torch.ones((predn.shape[0], 1)) * image_idx], dim=1) # append image_idx
        labelsn_with_path = torch.cat([labelsn, torch.ones((labelsn.shape[0], 1)) * image_idx], dim=1)  # append image_idx

        true_pos.append(predn_with_path[np.logical_and(preds_matched, preds_pos)])
        true_neg.append(predn_with_path[np.logical_and(preds_not_matched, preds_neg)])
        false_pos.append(predn_with_path[np.logical_and(preds_not_matched, preds_pos)])
        false_neg.append(labelsn_with_path[targets_not_matched])

    true_pos = torch.cat(true_pos)
    true_neg = torch.cat(true_neg)
    false_pos = torch.cat(false_pos)
    false_neg = torch.cat(false_neg)
    n_worst = 5

    fig = dict()
    for title, (gt_pos, assignment) in {
        "True Positive": (True, true_pos),
        "False Positive": (False, false_pos),
        "False Negative": (True, false_neg),
        "True Negative": (False, true_neg)  # TODO: Tricky interpretation c- for classification  & c+ for localization
    }.items():
        n_first = min(n_worst, len(assignment) // 2)
        assignment_lowest = assignment[np.argpartition(assignment[:, 4], +n_first)[:+n_first]]
        assignment_highest = assignment[np.argpartition(assignment[:, 4], -n_first)[-n_first:]]

        thumbnail_stack_lowest = [generate_crop(extra_stats, crop) for crop in assignment_lowest]
        thumbnail_stack_highest = [generate_crop(extra_stats, crop) for crop in assignment_highest]
        fig[title] = plot_crops(thumbnail_stack_lowest, thumbnail_stack_highest, title, gt_pos)
        fig[title].show()

    return fig
