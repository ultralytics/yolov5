# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset.

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_native, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """
    Saves detection results in txt format, including class, bounding box coordinates, and optionally confidence score.

    Args:
        predn (torch.Tensor): Tensor containing detection predictions in the format [x1, y1, x2, y2, confidence, class].
        save_conf (bool): If True, save confidence scores alongside other detection data.
        shape (tuple[int, int]): Original shape of the image (height, width).
        file (str | Path): Path to the file where the results will be saved.

    Returns:
        None

    Notes:
        The bounding box coordinates are saved in normalized xywh format (center_x, center_y, width, height), while the
        class is represented as an integer. If `save_conf` is True, confidence scores are also included in the output.

    Examples:
        ```python
        predn = torch.tensor([[100, 150, 200, 250, 0.98, 1]])  # Example prediction
        save_conf = True
        shape = (480, 640)  # Example image shape
        file = "output.txt"

        save_one_txt(predn, save_conf, shape, file)
        ```
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """
    Save detection results to a JSON dictionary, including bounding boxes, category IDs, scores, and segmentation masks.

    Args:
        predn (torch.Tensor): The predicted bounding boxes, scores, and class indices.
        jdict (list): The list to append the JSON formatted results.
        path (pathlib.Path): Path of the image file being processed.
        class_map (dict[int, int]): A dictionary mapping predicted class indices to COCO class indices.
        pred_masks (numpy.ndarray): The predicted segmentation masks.

    Returns:
        None. The function appends results to `jdict` in-place.

    Example JSON result:
        {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.

    Notes:
        - This function uses COCO's RLE (Run-Length Encoding) format to encode binary mask arrays for JSON serialization.
        - Ensure `predn` and `pred_masks` are in the correct format before calling this function.

    Example:
        ```python
        predn = torch.tensor([[100, 50, 200, 150, 0.8, 1], [120, 60, 220, 160, 0.75, 2]])
        jdict = []
        path = Path("/path/to/image.jpg")
        class_map = {0: 1, 1: 2}
        pred_masks = np.random.rand(256, 256, 2) > 0.5

        save_one_json(predn, jdict, path, class_map, pred_masks)
        ```
    """
    from pycocotools.mask import encode

    def single_encode(x):
        """Encodes binary mask arrays into RLE (Run-Length Encoding) format for JSON serialization."""
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        )


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return evaluation metrics for a batch of detections and labels, potentially incorporating segmentation masks.

    Args:
        detections (array[N, 6]): Array of detection results, where each row represents [x1, y1, x2, y2, confidence, class].
        labels (array[M, 5]): Array of ground truth labels, where each row represents [class, x1, y1, x2, y2].
        iouv (array): Array of IoU (Intersection over Union) thresholds to evaluate the detections against the labels.
        pred_masks (array, optional): Array of predicted segmentation masks for detected objects. Default is None.
        gt_masks (array, optional): Array of ground truth segmentation masks for labeled objects. Default is None.
        overlap (bool, optional): Whether to consider overlapping masks. Default is False.
        masks (bool, optional): Whether to process segmentation masks instead of bounding boxes. Default is False.

    Returns:
        correct (array[N, 10]): Boolean array indicating which detections are correct for each IoU level.

    Notes:
        - When `masks` is True, the function computes IoU based on the segmentation masks.
        - When `overlap` is True, label-specific indexing is applied to the ground truth masks to handle overlaps.

    Examples:
        ```python
        detections = np.array([[10, 20, 30, 40, 0.9, 1], [15, 25, 35, 45, 0.8, 2]])
        labels = np.array([[1, 10, 20, 30, 40], [2, 20, 30, 40, 50]])
        iouv = np.array([0.5, 0.75])
        correct = process_batch(detections, labels, iouv)
        ```
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
):
    """
    Validates a YOLOv5 segmentation model on specified dataset, producing metrics, plots, and optional JSON output.

    Args:
        data (str | dict): Path to dataset configuration file or dictionary defining data splits and other parameters.
        weights (str | list[str], optional): Path(s) to model weights file(s). Defaults to None.
        batch_size (int, optional): Batch size for inference. Defaults to 32.
        imgsz (int, optional): Inference image size in pixels. Defaults to 640.
        conf_thres (float, optional): Confidence threshold for predictions. Defaults to 0.001.
        iou_thres (float, optional): Intersection-over-union threshold for Non-Maximum Suppression (NMS). Defaults to 0.6.
        max_det (int, optional): Maximum number of detections per image. Defaults to 300.
        task (str, optional): Task type, can be 'train', 'val', 'test', 'speed', or 'study'. Defaults to 'val'.
        device (str, optional): Device for inference, e.g., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        workers (int, optional): Number of maximum dataloader workers (per RANK in DDP mode). Defaults to 8.
        single_cls (bool, optional): If True, treats the dataset as a single-class dataset. Defaults to False.
        augment (bool, optional): If True, performs augmented inference. Defaults to False.
        verbose (bool, optional): If True, enables verbose output. Defaults to False.
        save_txt (bool, optional): If True, saves results to *.txt files. Defaults to False.
        save_hybrid (bool, optional): If True, saves hybrid label+prediction results to *.txt files. Defaults to False.
        save_conf (bool, optional): If True, saves confidences in --save-txt labels. Defaults to False.
        save_json (bool, optional): If True, saves a COCO-JSON results file. Defaults to False.
        project (str | Path, optional): Project directory to save results to. Defaults to ROOT / "runs/val-seg".
        name (str, optional): Subdirectory name for this run's results. Defaults to "exp".
        exist_ok (bool, optional): If True, existing project/name is ok and does not increment. Defaults to False.
        half (bool, optional): If True, uses FP16 half-precision inference. Defaults to True.
        dnn (bool, optional): If True, uses OpenCV DNN for ONNX inference. Defaults to False.
        model (torch.nn.Module, optional): Pre-loaded model for validation. Defaults to None.
        dataloader (DataLoader, optional): Pre-loaded dataloader for validation. Defaults to None.
        save_dir (Path, optional): Directory to save results. Defaults to Path("").
        plots (bool, optional): If True, generates plots of predictions and metrics. Defaults to True.
        overlap (bool, optional): If True, enables segmentation mask overlap during processing. Defaults to False.
        mask_downsample_ratio (int, optional): Downsample ratio for segmentation masks. Defaults to 1.
        compute_loss (Callable, optional): Function to compute loss. Defaults to None.
        callbacks (Callbacks, optional): Custom callbacks for validation events. Defaults to Callbacks().

    Returns:
        tuple: Validation metrics including precision and recall for bounding boxes and masks (mp_bbox, mr_bbox, map50_bbox,
        map_bbox, mp_mask, mr_mask, map50_mask, map_mask).
    """
    if save_json:
        check_requirements("pycocotools>=2.0.6")
        process = process_mask_native  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
        nm = de_parallel(model).model[-1].nm  # number of masks
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio,
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 10) % (
        "Class",
        "Images",
        "Instances",
        "Box(P",
        "R",
        "mAP50",
        "mAP50-95)",
        "Mask(P",
        "R",
        "mAP50",
        "mAP50-95)",
    )
    dt = Profile(device=device), Profile(device=device), Profile(device=device)
    metrics = Metrics()
    loss = torch.zeros(4, device=device)
    jdict, stats = [], []
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            masks = masks.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, protos, train_out = model(im) if compute_loss else (*model(im, augment=augment)[:2], None)

        # Loss
        if compute_loss:
            loss += compute_loss((train_out, protos), targets, masks)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det, nm=nm
            )

        # Metrics
        plot_masks = []  # masks for plotting
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15])  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                pred_masks = scale_image(
                    im[si].shape[1:], pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1]
                )
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)
            plot_images_and_masks(
                im,
                output_to_target(preds, max_det=15),
                plot_masks,
                paths,
                save_dir / f"val_batch{batch_i}_pred.jpg",
                names,
            )  # pred

        # callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            results = []
            for eval in COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm"):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t


def parse_opt():
    """
    Parses command line arguments for configuring YOLOv5 options including dataset path, model weights, batch size, and
    inference settings.

    Args:
        --data (str): Path to the dataset YAML file. Default is 'data/coco128-seg.yaml'.
        --weights (list[str]): Model path(s). Default is 'yolov5s-seg.pt'.
        --batch-size (int): Batch size for data loading. Default is 32.
        --imgsz (int): Inference image size in pixels. Default is 640.
        --conf-thres (float): Confidence threshold for object detection. Default is 0.001.
        --iou-thres (float): IoU threshold for non-max suppression. Default is 0.6.
        --max-det (int): Maximum number of detections per image. Default is 300.
        --task (str): Task type, can be 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        --device (str): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Default is ''.
        --workers (int): Maximum number of dataloader workers per rank in DDP mode. Default is 8.
        --single-cls (bool): Treat dataset as single-class. Default is False.
        --augment (bool): Apply augmentation during inference. Default is False.
        --verbose (bool): Report mean Average Precision (mAP) by class. Default is False.
        --save-txt (bool): Save results to text files (*.txt). Default is False.
        --save-hybrid (bool): Save hybrid results to text files (*.txt), including both labels and predictions. Default is False.
        --save-conf (bool): Save confidences in text labels. Default is False.
        --save-json (bool): Save results as a COCO-JSON file. Default is False.
        --project (str): Project directory to save results. Default is 'runs/val-seg'.
        --name (str): Experiment name to save results. Default is 'exp'.
        --exist-ok (bool): Allow existing project/name, do not increment. Default is False.
        --half (bool): Use FP16 half-precision inference. Default is False.
        --dnn (bool): Use OpenCV DNN for ONNX inference. Default is False.

    Returns:
        argparse.Namespace: Parsed command line arguments populated into a Namespace object.

    Notes:
        - Args with multiple names like '--imgsz', '--img', '--img-size' all map to the same attribute.
        - Automatically checks for valid YAML file structure in the dataset argument.
        - Combines additional argument checks, like setting save_txt to True when save_hybrid is enabled.
        - Ensures compatibility with various inference backends like PyTorch, ONNX, TensorFlow, etc.

    Example:
        ```python
        opt = parse_opt()
        print(opt.data, opt.weights, opt.batch_size)
        ```

    Links:
        - Official repository: https://github.com/ultralytics/ultralytics
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 tasks including training, validation, testing, speed benchmarking, and study analysis with
    configurable options.

    Args:
      opt (argparse.Namespace): Command line arguments parsed by `parse_opt()` function.
        - data (str): Path to dataset YAML file.
        - weights (list[str]): Path(s) to model weights.
        - batch_size (int): Batch size for inference.
        - imgsz (int): Inference image size in pixels.
        - conf_thres (float): Confidence threshold for predictions.
        - iou_thres (float): IoU threshold for non-maximum suppression.
        - max_det (int): Maximum number of detections per image.
        - task (str): Task to be executed; options are "train", "val", "test", "speed", or "study".
        - device (str): Device to run on; CUDA device indexes or "cpu".
        - workers (int): Number of workers for data loading per RANK in DDP mode.
        - single_cls (bool): Flag to treat dataset as single-class.
        - augment (bool): Flag for augmented inference.
        - verbose (bool): Flag for verbose output, including mAP by class.
        - save_txt (bool): Flag to save results to *.txt files.
        - save_hybrid (bool): Flag to save hybrid results (label + prediction) to *.txt.
        - save_conf (bool): Flag to include confidences in saved *.txt results.
        - save_json (bool): Flag to save results as COCO-JSON.
        - project (str): Directory to save results.
        - name (str): Name of the run; results will be saved to 'project/name'.
        - exist_ok (bool): Flag to allow overwriting existing 'project/name' without incrementing.
        - half (bool): Flag to use FP16 half-precision inference, if supported by the hardware.
        - dnn (bool): Flag to use OpenCV DNN for ONNX inference.

    Returns:
      None

    Examples:
      ```python
      if __name__ == "__main__":
          opt = parse_opt()
          main(opt)
      ```

    Notes:
      Make sure to specify `task` appropriately as per the intended YOLOv5 operation: "train", "val", "test", "speed", or "study".
      For speed benchmarking, set `task` to "speed" and adjust `conf_thres` and `iou_thres` as necessary. For study analysis, set `task` to "study" and run with various `imgsz` values.
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
