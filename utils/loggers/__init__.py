# YOLOv5 experiment logging utils

import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr
from utils.wandb_logging.wandb_utils import WandbLogger


def init_loggers(save_dir, weights, opt, hyp, data_dict, logger, include=('tensorboard', 'wandb')):
    # Initialize loggers at train start
    loggers = {'wandb': None, 'tb': None}  # loggers dict
    project = save_dir.parent

    # TensorBoard
    if 'tensorboard' in include:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {project}', view at http://localhost:6006/")
        loggers['tb'] = SummaryWriter(str(save_dir))

    # W&B
    if 'wandb' in include:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if opt.resume else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb

    return loggers
