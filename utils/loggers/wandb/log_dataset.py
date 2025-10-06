import argparse

from wandb_utils import WandbLogger

from utils.general import LOGGER

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Creation')  # TODO: return value unused
    if not logger.wandb:
        LOGGER.info("install wandb using `pip install wandb` to log the dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', type=str, default='log dataset', help='name of W&B run')

    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
