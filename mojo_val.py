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


def draw_targets(frame, bboxes, matches, mask, rect=False):
    """
    Draw part of targets and predictions in several colors onto an image
    Arguments:
        frame (Array[H, W, C]), images to be drawn
        bboxes (Array[N, 4]), x1, y1, x2, y2, coordinates of the target
        matches (Array[N,]), x1, y1, x2, y2 bool vector to show in red or green color
        mask (Array[N]), x1, y1, x2, y2 bool vector to show or not the prediction if > conf_thresh
    Returns:
        None since frame has been drawn inplace
    """
    assert bboxes.shape[0] == matches.shape[0] == mask.shape[0]
    clip_coords(bboxes, frame.shape)
    bboxes = bboxes.astype(int)
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


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y


def generate_crop(extra_stats, predn_with_path, BGR=False):
    """
    Draw part of targets and predictions in several colors onto an image
      Arguments:
        extra_stats Array[size of val dataset, 6]
            List of path_to_image, *whatever
            to be used only for finding the image path linked with the prediction
        predn_with_path Array[N, 7] x1, y1, x2, y2, *whatever, path_idx
            Prediction with the x,y coordinates and the path idx for the image to cut a crop out of it
    Returns:
        image Array[gain * (y2 - y1), gain * (x2 - x1), C] Crop of the prediction in RGB with a gain
    """
    bbox = predn_with_path[:4]
    frame_idx = int(predn_with_path[-1])  # paths idx should be at the end
    frame_path = extra_stats[frame_idx][0]
    frame = cv2.imread(str(frame_path), 1)
    bbox = bbox.unsqueeze(0)
    clip_coords(bbox, frame.shape)
    bbox = bbox[0]
    bbox = [int(b) for b in bbox]
    frame_crop = frame[bbox[1]:bbox[3], int(bbox[0]):bbox[2], ::(1 if BGR else -1)]
    return frame_crop


def plot_crops(crops_low, assignment_low, crops_high, assignment_high, title, gt_pos, *, n_cols):
    """
    Draw part of targets and predictions in several colors onto an image
    Arguments:
        crops_low List[Image] (Cropped images of target or predictions to be drawn with low conf
        assignment_low Predictions or labels associated with crop_low to retrieve confidence value
        crops_high List[Image] Cropped images of target or predictions to be drawn with high conf
        assignment_high Predictions or labels associated with crop_high to retrieve confidence value
        title str Name of the category
        gt_pos bool Whether to draw a symbol of a good or bad predictions or labels given its category
    Returns:
        fig go.Figure Figure of a grid of crops
    """
    titles_low = [f"{'💔' if gt_pos else '💚' } Low conf (c={c:.2f})" for c in assignment_low[:, 4]]
    titles_high = [f"{'💚' if gt_pos else '💔' } High conf (c={c:.2f})" for c in assignment_high[:, 4]]
    crops = crops_low + crops_high
    n_rows_low = len(crops_low) // n_cols + 1
    n_rows_high = len(crops_high) // n_cols + 1
    titles = titles_low + [None for _ in range(n_cols - len(titles_low) % n_cols)] + titles_high
    fig = make_subplots(
        rows=n_rows_low + n_rows_high,
        cols=n_cols,
        subplot_titles=titles
    )

    for n, image in enumerate(crops_high):
        fig.add_trace(px.imshow(image).data[0], row=int(n / n_cols) + 1, col=n % n_cols + 1)

    for n, image in enumerate(crops_low):
        fig.add_trace(px.imshow(image).data[0], row=n_rows_high + int(n / n_cols) + 1, col=n % n_cols + 1)

    if len(crops):
        layout = px.imshow(crops[0], color_continuous_scale='gray').layout
        fig.layout.coloraxis = layout.coloraxis
        fig.update_xaxes(**layout.xaxis.to_plotly_json())
        fig.update_yaxes(**layout.yaxis.to_plotly_json())
    fig.layout.title = title
    return fig


def compute_and_plot_predictions_and_labels(extra_stats, threshold):
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

    debug_frames = []
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
        # Get pos, negn matched and non matched to compute and show FP/FN/TP/TN
        preds_matched = preds_matched[:, 0] # iou = 0.5
        preds_not_matched = np.logical_not(preds_matched)
        targets_not_matched = np.logical_not(targets_matched)
        preds_pos = predn[:, 4] >= threshold
        preds_neg = predn[:, 4] < threshold

        # Draw one image
        if image_idx in [0, 1, 2]:
            preds_pos = predn[:, 4] >= threshold
            preds_neg = predn[:, 4] < threshold

            frame = cv2.imread(str(path), 1)

            # Draw targets that are missed or not
            draw_targets(frame, labelsn.numpy(), np.logical_not(targets_not_matched.numpy()), np.ones(labels.shape[0]))
            # Draw wrong and good preds that are above a certain threshold
            draw_targets(frame, predn[:, :4].numpy(), preds_matched.numpy(), preds_pos, rect=True)

            debug_frames.append(frame)

        predn_with_path = torch.cat([predn, torch.ones((predn.shape[0], 1)) * image_idx], dim=1)  # append image_idx
        labelsn_with_path = torch.cat([labelsn, torch.ones((labelsn.shape[0], 1)) * image_idx], dim=1)  # append image_idx

        true_pos.append(predn_with_path[np.logical_and(preds_matched, preds_pos)])
        true_neg.append(predn_with_path[np.logical_and(preds_not_matched, preds_neg)])
        false_pos.append(predn_with_path[np.logical_and(preds_not_matched, preds_pos)])
        false_neg.append(labelsn_with_path[targets_not_matched])

    true_pos = torch.cat(true_pos)
    true_neg = torch.cat(true_neg)
    false_pos = torch.cat(false_pos)
    false_neg = torch.cat(false_neg)

    fig = dict()
    for image_idx, frame in enumerate(debug_frames):
        fig[f"Debug Image (id: {image_idx})"] = px.imshow(frame)

    fig.update(plot_predictions_and_labels(true_pos, true_neg, false_pos, false_neg, extra_stats, debug_frames))

    return fig


def plot_predictions_and_labels(true_pos, true_neg, false_pos, false_neg, extra_stats, debug_frames):
    """
    Plots predictions and labels based on everything
    Arguments:
    Returns:
        fig Dict[go.Figure] Dictionary of debug images and good/bad predictions labels as cropped images grid
    """
    n_crops_displayed = 10
    fig = dict()
    for image_idx, frame in enumerate(debug_frames):
        fig[f"Debug Image (id: {image_idx})"] = px.imshow(frame)

    for title, (gt_pos, assignment) in {
        "True Positive": (True, true_pos),
        "False Positive": (False, false_pos),
        "False Negative": (True, false_neg),
        "True Negative": (False, true_neg)  # TODO: Tricky interpretation c- for classification & c+ for localization
    }.items():
        n_first = min(n_crops_displayed, len(assignment) // 2)
        confidence = assignment[:, 4]
        assignment_lowest_idx = np.argpartition(confidence, +n_first)[:+n_first]
        assignment_highest_idx = np.argpartition(confidence, -n_first)[-n_first:]
        assignment_low = assignment[assignment_lowest_idx]
        assignment_high = assignment[assignment_highest_idx]

        crops_low = [generate_crop(extra_stats, pred) for pred in assignment_low]
        crops_high = [generate_crop(extra_stats, pred) for pred in assignment_high]
        fig[title] = plot_crops(crops_low, assignment_low, crops_high, assignment_high, title, gt_pos, n_cols=5)

    return fig
