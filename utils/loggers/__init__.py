# YOLOv5 experiment logging utils

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr
from utils.wandb_logging.wandb_utils import WandbLogger

LOGGERS = ('tb', 'wandb')  # default logger list


def init_loggers():
    # Initialize empty logger dictionary
    return {k: None for k in LOGGERS}


def start_loggers(save_dir, weights, opt, hyp, data_dict, logger, include=LOGGERS):
    # Start loggers at train start
    loggers = init_loggers()
    project = save_dir.parent

    # TensorBoard
    if 'tb' in include and not opt.evolve:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {project}', view at http://localhost:6006/")
        loggers['tb'] = SummaryWriter(str(save_dir))

    # W&B
    if 'wandb' in include:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if opt.resume else None
        loggers['wandb'] = WandbLogger(opt, save_dir.stem, run_id, data_dict)

    return loggers
