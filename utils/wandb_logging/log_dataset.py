import argparse
import os
from pathlib import Path

import torch
import yaml

from wandb_utils import WandbLogger
from utils.datasets import create_dataloader, LoadImagesAndLabels
from utils.general import check_dataset, colorstr
from utils.torch_utils import torch_distributed_zero_first

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    wandb_logger = WandbLogger(opt, '', None, data_dict, job_type='create_dataset')
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])
    trainset = LoadImagesAndLabels(data_dict['train'], rect=opt.rect)
    testset = LoadImagesAndLabels(data_dict['val'], rect=opt.rect)
    names_to_ids = {k: v for k, v in enumerate(names)}
    wandb_logger.log_dataset_artifact(trainset, names_to_ids, name='train')
    wandb_logger.log_dataset_artifact(testset, names_to_ids, name='val')
    # Update/Create new config file with links to artifact
    data_dict['train'] = WANDB_ARTIFACT_PREFIX + str(Path(opt.project) / 'train')
    data_dict['val'] = WANDB_ARTIFACT_PREFIX + str(Path(opt.project) / 'val')
    ouput_data_config = opt.data if opt.overwrite_config else opt.data.replace('.', '_wandb.')
    data_dict.pop('download', None)  # Don't download the original dataset. Use artifacts
    with open(ouput_data_config, 'w') as fp:
        yaml.dump(data_dict, fp)
    print("New Config file => ", ouput_data_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--project', type=str, default='yolov5', help='name of W&B Project')
    parser.add_argument('--overwrite_config', action='store_true', help='replace the origin data config file')
    opt = parser.parse_args()

    create_dataset_artifact(opt)
