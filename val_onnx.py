# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to Validate YOLO ONNX models

##########
Example:
1) Validate a 98% Sparse YOLO model on COCO
$python val_onnx.py

2) Validation a YOLO model stub on COCO
$python val_onnx.py \
    --model_path \
    zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98
"""

from sparseml.onnx.utils import get_tensor_dim_shape


try:
    from deepsparse import Pipeline

except Exception as deepsparse_error:
    raise RuntimeError(
        "Unable to import Pipeline from deepsparse"
        f"DeepSparse>=13 must be installed {deepsparse_error}"
    )


import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from tqdm import tqdm

from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (
    LOGGER,
    ROOT,
    box_iou,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    print_args,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import plot_val_study
from utils.torch_utils import select_device, time_sync


DEEPSPARSE = "deepsparse"
ONNX_RUNTIME = "onnxruntime"

FILE = Path(__file__).resolve()
LOCAL_ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(LOCAL_ROOT) not in sys.path:
    sys.path.append(str(LOCAL_ROOT))  # add ROOT to PATH
LOCAL_ROOT = Path(os.path.relpath(LOCAL_ROOT, Path.cwd()))  # relative


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (
            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        )  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18,
    # "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(
        detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
    )
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where(
        (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
    )  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = (
            torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        )  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def get_stride(yolo_pipeline: Pipeline) -> int:
    """
    Infer max stride from pipeline

    :param yolo_pipeline: pipeline to infer the max stride of
    """
    model = onnx.load(yolo_pipeline.onnx_file_path)
    image_size = get_tensor_dim_shape(model.graph.input[0], 2)
    if not image_size:
        image_size = yolo_pipeline.image_size or 640
        if not isinstance(image_size, int):
            image_size = image_size[0]

    grid_shapes = list(
        get_tensor_dim_shape(model.graph.output[index], 2)
        for index in range(1, len(model.graph.output))
    )

    def _infer_grid_shapes():
        # build fake input
        input_shape = [
            yolo_pipeline.engine.batch_size,
            get_tensor_dim_shape(model.graph.input[0], 1),
            image_size,
            image_size
        ]
        fake_input = np.random.randn(*input_shape).astype(
            np.uint8 if yolo_pipeline.is_quantized else np.float32
        )

        # run sample forward pass and get grid shapes from output size
        fake_outputs = yolo_pipeline.engine([fake_input])[1:]  # skip first output
        return [output.shape[2] for output in fake_outputs]

    if any(not grid_shape for grid_shape in grid_shapes):
        # unable to get static output shape, infer from forward pass
        grid_shapes = _infer_grid_shapes()

    strides = (image_size // grid_shape for grid_shape in grid_shapes)
    return max(strides)


@torch.no_grad()
def run(
    data,
    data_path='', # optional data path to overwrite one written in .yaml data file
    model_path=None,  # model.onnx path/ SparseZoo stub
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
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
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    engine=DEEPSPARSE,
    num_cores=None,
):
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load pipeline
    yolo_pipeline = Pipeline.create(
        task="yolo",
        model_path=model_path,
        class_names="coco",
        engine_type=engine,
        num_cores=num_cores,
        batch_size=batch_size,
    )

    stride = get_stride(yolo_pipeline)

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Data
    data = check_dataset(data, data_path)  # check

    # Configure
    # Note: Only coco validation data is supported for now
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(
        "coco/val2017.txt"
    )  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    pad = 0.0 if task in ("speed", "benchmark") else 0.5
    rect = False if task == "benchmark" else None  # square inference for benchmarks
    task = (
        task if task in ("train", "val", "test") else "val"
    )  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=1,
        # workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    # Note: Only coco validation data is supported for now
    names = yolo_pipeline.class_names

    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%20s" + "%11s" * 6) % (
        "Class",
        "Images",
        "Labels",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )
    dt, p, r, f1, mp, mr, map50, map = (
        [0.0, 0.0, 0.0],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(
        dataloader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out = yolo_pipeline(
            images=[im.numpy()], iou_thres=iou_thres, conf_thres=conf_thres
        )

        # inference, loss outputs
        dt[1] += time_sync() - t2

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height))  # to pixels
        t3 = time_sync()
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            pred = torch.Tensor(pred.predictions)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0

            predn = pred.clone()
            scale_coords(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                iouv = iouv.cpu()
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(
                    predn,
                    save_conf,
                    shape,
                    file=save_dir / "labels" / (path.stem + ".txt"),
                )
            if save_json:
                save_one_json(
                    predn, jdict, path, class_map
                )  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=nc
        )  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image

    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per "
        f"image at shape {shape}" % t
    )

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end")

    # Save JSON
    if save_json and len(jdict):
        w = ""
        anno_json = str(
            Path(data.get("path", "../coco")) / "annotations/instances_val2017.json"
        )  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb # noqa
            check_requirements(["pycocotools"])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [
                    int(Path(x).stem) for x in dataloader.dataset.im_files
                ]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    s = (
        f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to "
        f"{save_dir / 'labels'}"
        if save_txt
        else ""
    )
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map

    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path"
    )
    default_yolo_stub = (
        "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        type=str,
        default=default_yolo_stub,
        help="model.onnx path or a YOLO SparseZoo model stub",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--task", default="val", help="train, val, test, speed or study"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid",
        action="store_true",
        help="save label+prediction hybrid results to *.txt",
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="save a COCO-JSON results file"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/val", help="save to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=[DEEPSPARSE, ONNX_RUNTIME],
        help="The engine to use for validation, choose b/w deepsparse and onnxruntime",
        default=DEEPSPARSE,
    )
    parser.add_argument(
        "--num_cores",
        "--num_cores",
        type=int,
        default=None,
        help="Number of cores to use for Validation. "
        "If None, all available cores will be used.",
    )
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt=None):
    if opt is None:
        opt = parse_opt()
    check_requirements(
        requirements=LOCAL_ROOT / "requirements.txt", exclude=("tensorboard", "thop")
    )

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(
                f"WARNING: confidence threshold {opt.conf_thres} >> 0.001 will "
                f"produce invalid mAP values."
            )
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt... # noqa
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt... # noqa
            for opt.weights in weights:
                # filename to save to
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"
                x, y = (
                    list(range(256, 1536 + 128, 128)),
                    [],
                )  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            os.system("zip -r study.zip study_*.txt")
            plot_val_study(x=x)  # plot

def val_onnx_run(**kwargs):
    opt = parse_opt()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
