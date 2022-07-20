#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import argparse
import os
import subprocess
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--weights', type=str, required=True, help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    return parser.parse_args()


def run_validation_on_dataset(dataset_path, argument_list):
    name = os.path.splitext(os.path.basename(dataset_path))[0]
    command = ['python3', 'val.py', '--data', dataset_path, '--name', name] + argument_list
    print(command)
    subprocess.run(command)


def get_constant_commands_from_args(args):
    arg_dict = vars(args)
    del arg_dict['data_folder']
    arg_list = []
    for arg, val in arg_dict.items():
        arg_key = f"--{arg}".replace('_', '-')
        if type(val) == bool:
            if val:
                arg_list.append(arg_key)
        else:
            val_key = str(val)
            arg_list.extend([arg_key, val_key])
    return arg_list


def get_datasets_from_folder(folder_path):
    datasets = []
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isfile(item_path) and '.yaml' in item_path:
            datasets.append(item_path)
    return datasets


def main(args):
    dataset_filepaths = get_datasets_from_folder(args.data_folder)
    formatted_arg_list = get_constant_commands_from_args(args)
    for filepath in dataset_filepaths:
        run_validation_on_dataset(filepath, formatted_arg_list)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)