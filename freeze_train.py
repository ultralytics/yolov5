"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.plots import plot_evolve, plot_labels
from utils.metrics import fitness
from utils.loss import ComputeLoss
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loggers import Loggers
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.downloads import attempt_download
from utils.datasets import create_dataloader
from utils.callbacks import Callbacks
from utils.autobatch import check_train_batch_size
from utils.autoanchor import check_anchors
from models.yolo import Model
from models.experimental import attempt_load
import train
import val  # for end-of-epoch mAP
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/freeze', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze-plan', type=str, default='', help='Specify the freeze training plan yaml file')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    # TODO resume mode
    # if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
    #     ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
    #     assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
    #     with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
    #         opt = argparse.Namespace(**yaml.safe_load(f))  # replace
    #     opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    #     LOGGER.info(f'Resuming training from {ckpt}')
    if 1:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    # if LOCAL_RANK != -1:
    #     assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
    #     assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
    #     assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
    #     assert not opt.evolve, '--evolve argument is not compatible with DDP training'
    #     torch.cuda.set_device(LOCAL_RANK)
    #     device = torch.device('cuda', LOCAL_RANK)
    #     dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Freeze train
    parent_save_dir = opt.save_dir
    with open(opt.freeze_plan, errors='freeze_plan cannot find!') as f:
        freeze_plan = yaml.safe_load(f)

    for arg in ['evolve', 'resume']:
        if opt.__getattribute__('resume'):
            opt.__setattr__(arg, None)
        LOGGER.info(f'Currently option --{arg} is not support in freeze training mode')
    LOGGER.info('Starting freeze training!')

    for i, plan in enumerate(freeze_plan['train']):
        print(plan,'\n')
        assert len(plan) == 2, "ERROR: Please check your freeze plan format! It should be [strategy, epoch]"
        strategy = plan[0]
        strategy_epochs = plan[1]
        for strategy_epoch in range(strategy_epochs):
            for step in freeze_plan[strategy]:
                if len(step) == 2:
                    opt.freeze_type = step[0]
                    opt.freeze = ''
                    opt.epochs = step[1]
                    LOGGER.info(f'Currently training strategy epoch {strategy_epoch}, step {i} of strategy {strategy}; training without freeze, \
                        Epoch is {opt.epochs}.')
                elif len(step) == 3:
                    opt.freeze_type = step[0]
                    opt.freeze = step[1]
                    opt.epochs = step[2]
                    LOGGER.info(f'Currently training strategy epoch {strategy_epoch}, step {i} of strategy {strategy}; Freeze type is {opt.freeze}, \
                        Epoch is {opt.epochs}, Freeze layers {opt.freeze}')
                else:
                    raise Exception(
                        f'ERROR: format of each step in strategy should be [layer select type, epoch] or [layer select type, freeze layer, epoch]')
                opt.save_dir = str(increment_path(Path(parent_save_dir) / opt.name, exist_ok=opt.exist_ok))
                train.train(opt.hyp, opt, device, callbacks)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
