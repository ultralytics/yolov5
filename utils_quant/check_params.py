import os
from pathlib import Path
import logging

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml

logger = logging.getLogger(__name__)

try:
    from train import train
    import test  # import test.py to get mAP after each epoch
    from utils.general import check_file, colorstr
    from utils.torch_utils import select_device
except Exception as e:
    print(repr(e))


def check_and_set_params(opt):
    """
    i. Check the validity of parameters
    ii. Convert for compatibility with yolov5 parameters, and supplement the necessasy parameters
    """
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Conversion and supplement for compatibility with yolov5 parameters
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.epochs = opt.num_finetune_epochs
    opt.batch_size = opt.batch_size_train
    opt.weights = opt.ckpt_path
    opt.total_batch_size = opt.batch_size_train
    opt.project = opt.out_dir  # output folder
    opt.evolve = False
    opt.resume = False
    opt.single_cls = False
    opt.adam = False
    opt.linear_lr = False
    opt.sync_bn = False
    opt.cache_images = False
    opt.image_weights = False
    opt.rect = False
    opt.workers = 8             # maximum number of dataloader workers
    opt.quad = False            # quad dataloader
    opt.noautoanchor = False    # disable autoanchor check
    opt.label_smoothing = 0.0   # default=0.0, Label smoothing epsilon
    opt.multi_scale = False     # vary img-size +/- 50%
    opt.notest = False          # only test final epoch
    opt.name = 'exp'            # save to project/name
    opt.bucket = ''             # gsutil bucket
    opt.nosave = False          # only save final checkpoint
    opt.conf_thres = 0.001      # default=0.001, help='object confidence threshold'
    opt.iou_thres = 0.6         # default=0.6, help='IOU threshold for NMS'
    opt.exist_ok = False        # action='store_true', help='existing project/name ok, do not increment'
    opt.save_txt = False        # action='store_true', help='save results to *.txt')
    opt.save_hybrid = False     # action='store_true', help='save label+prediction hybrid results to *.txt')
    opt.save_conf = False       # action='store_true', help='save confidences in --save-txt labels'
    opt.save_json = True        # save a cocoapi-compatible JSON results file

    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))                # extend to 2 sizes (train, test)
    opt.save_dir = Path(opt.out_dir)

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    logger.info(opt)
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.out_dir}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

    return hyp, opt, device, tb_writer