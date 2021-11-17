import numpy as np
import torch
import cv2
from aisa_utils.dl_utils.utils import plot_object_count_difference_ridgeline
from torchvision.ops import box_iou

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
    true_positive = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    false_negative = torch.ones(labels.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
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
        false_negative = ~matches_by_iou
        true_positive[matches[:, 1].long()] = matches_by_iou
    return true_positive, false_negative


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
    frame_idx = int(predn_with_path[6])
    frame_path = extra_stats[frame_idx][0]
    frame = cv2.imread(str(frame_path), 1)

    frame_crop = save_one_box(crop_box, frame, BGR=True, save=False)
    return frame_crop


def process_fp_fn(extra_stats):
    iouv = torch.linspace(0.5, 0.95, 10)
    single_cls = True
    niou = len(iouv)
    for image_idx, (path, pred, labels, img1_shape, img0_shape, ratio_pad) in enumerate(extra_stats):
        high_conf_fp = torch.zeros(0, pred.shape[1])
        nl = len(labels)

        if len(pred) == 0:
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_coords(img1_shape, predn[:, :4], img0_shape, ratio_pad)  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img1_shape, tbox, img0_shape, ratio_pad)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct, false_positive = process_batch_with_missed_labels(predn, labelsn, iouv)

        else:
            correct, false_positive = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

        targets_miss = false_positive[:, 0]  # iouv == 0.5
        pred_correct = correct[:, 0].numpy()
        labelsn = labelsn[:, 1:] == 0.5
        frame = cv2.imread(str(path), 1)

        # torch.stack(torch.ones(pred.shape[0]) * image_idx, pred, axis=0)
        if False:
            # Draw targets that are missed or not
            draw_targets(frame, labelsn.numpy(), np.logical_not(targets_miss.numpy()), np.ones(labels.shape[0]),  rect=True)
            # Draw wrong and good preds that are above a certain threshold
            draw_targets(frame, predn[:, :4].numpy(), pred_correct, predn[:, 4] > 0.1)

            cv2.imshow("d", frame)
            cv2.waitKey()
            cv2.destroyAllWindows()

        predn_with_path = torch.cat([predn, torch.ones((predn.shape[0], 1)) * image_idx], dim=1) # append image_idx
        true_pos = predn_with_path[pred_correct]
        false_pos = predn_with_path[np.logical_not(pred_correct)]
        false_neg = labelsn[targets_miss][:, 1]

        if high_conf_fp.shape[0] != 0:
            high_conf_fp = torch.cat([high_conf_fp, false_pos])
            high_conf_fp = high_conf_fp[torch.argsort(high_conf_fp[:, 4], descending=True)]
            high_conf_fp = high_conf_fp[:5]
        else:
            high_conf_fp = false_pos

        from plotly.subplots import make_subplots
        import plotly.express as px
        crop_fp = [generate_crop(extra_stats, crop) for crop in high_conf_fp]

        fig = make_subplots(rows=len(crop_fp) // 5 + 1, cols=5)

        for n, image in enumerate(crop_fp):
            fig.add_trace(px.imshow(255 - image).data[0], row=int(n / 5) + 1, col=n % 5 + 1)

        # the layout gets lost, so we have to carry it over - but we cannot simply do
        # fig.layout = layout since the layout has to be slightly different for subplots
        # fig.layout.yaxis in a subplot refers only to the first axis for example
        # update_yaxes updates *all* axis on the other hand
        layout = px.imshow(crop_fp[0], color_continuous_scale='gray').layout
        fig.layout.coloraxis = layout.coloraxis
        fig.update_xaxes(**layout.xaxis.to_plotly_json())
        fig.update_yaxes(**layout.yaxis.to_plotly_json())
        fig.show()