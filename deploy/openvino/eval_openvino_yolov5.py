#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import json
import logging
import sys
import time
import torch
from argparse import ArgumentParser, SUPPRESS
import numpy as np

from tqdm import tqdm
from pathlib import Path
sys.path.append("../../")

from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_requirements, box_iou, non_max_suppression, \
    scale_coords, xyxy2xywh, xywh2xyxy, increment_path


from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument('--data', type=str, default='../datasets/coco/val2017.txt', help='val2017.txt path')
    args.add_argument("--batch-size", default=1, type=int, help="batch size.")
    args.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="cpu", type=str)
    args.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    args.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    args.add_argument("--is-async", action="store_true", help="if or not use asynchronous inference.")
    args.add_argument("--num-requests", default=None, type=int,
                      help="if async, indicates the maximum number of requests received at the same time")
    args.add_argument('--name', default='exp', help='save to project/name')
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    return parser


def save_one_json(predn, jdict, path, class_map):
    """
    Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def main():
    args = build_argparser().parse_args()

    device = args.device

    # Directories
    save_dir = increment_path(Path('runs/val') / args.name)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    pad = 0.5

    gs = 32
    dataloader = create_dataloader(args.data, args.imgsz, args.batch_size, gs, pad=pad, rect=False)[0]

    infer_time, seen = [0.0, 0.0, 0.0], 0
    jdict, stats = [], []
    class_map = coco80_to_coco91_class()
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    model = args.model
    weights = Path(model).with_suffix('.bin')
    log.info(f"Loading network:\n\t{model}, {weights}")
    net = ie.read_network(model=model, weights=weights)

    # ---------------------------------- 3. Synchronous and asynchronous inference -------------------------------------
    if args.is_async:
        exec_net = ie.load_network(network=net, device_name="CPU", num_requests=args.num_requests)
    else:
        exec_net = ie.load_network(network=net, device_name="CPU")

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.input_info))

    #  Defaulf batch_size is 1
    net.batch_size = args.batch_size

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        t1 = time.time()
        img = img.float().to(device, non_blocking=True)

        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # tensor.cuda --> numpy.cpu
        img = img.numpy()
        t2 = time.time()
        infer_time[0] += t2 - t1

        # Run model
        outputs = exec_net.infer(inputs={input_blob: img})
        infer_time[1] += time.time() - t2

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels

        t3 = time.time()
        out = non_max_suppression(torch.from_numpy(outputs['Concat_358']), args.conf_thres,
                                  args.iou_thres, multi_label=True)
        infer_time[2] += time.time() - t3

        for si, pred in enumerate(out):
            seen += 1
            labels = targets[targets[:, 0] == si, 1:]

            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

    # Print inference times
    time_per_image = tuple(x / seen * 1E3 for x in infer_time)  # speeds per image
    shape = (args.batch_size, 3, args.imgsz, args.imgsz)
    print(f'Speed: {time_per_image[0]:.1f}ms pre-process, {time_per_image[1]:.1f}ms inference, '
          f'{time_per_image[2]:.1f}ms NMS per image at shape {shape}')

    if len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(
            '../datasets/coco/annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        check_requirements(['pycocotools'])
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)


if __name__ == "__main__":
    main()