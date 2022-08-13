# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - train:
    $ python classifier.py --model yolov5s --data cifar100 --epochs 5 --img 224 --batch 128
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classifier.py --model yolov5s --data imagenet --epochs 5 --img 224 --device 4,5,6,7

Usage - inference:
    from classifier import *

    model = torch.load('path/to/best.pt', map_location=torch.device('cpu'))['model'].float()
    files = Path('../datasets/mnist/test/7').glob('*.png')  # images from dir
    for f in list(files)[:10]:  # first 10 images
        classify(model, size=128, file=f)
"""

import argparse
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import denormalize, normalize
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, check_git_status, check_requirements, colorstr, download, increment_path, init_seeds,
                           print_args)
from utils.loggers import GenericLogger
from utils.torch_utils import (ModelEMA, model_info, select_device, smart_DDP, smart_hub_load, smart_inference_mode,
                               smart_optimizer, torch_distributed_zero_first, update_classifier_model)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        Path(opt.save_dir), Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dir = data if data.is_dir() else (FILE.parents[1] / 'datasets' / data)
        if not data_dir.is_dir():
            LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
            t = time.time()
            if data == 'imagenet':
                subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
            else:
                url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=opt.cache,
                                                   rank=LOCAL_RANK,
                                                   workers=nw)

    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=opt.cache,
                                                      rank=-1,
                                                      workers=nw)

    # Initialize
    names = trainloader.dataset.classes  # class names
    LOGGER.info(f'Training {opt.model} on {data} dataset with {nc} classes...')

    # Model
    repo1, repo2 = 'ultralytics/yolov5', 'pytorch/vision'
    with torch_distributed_zero_first(LOCAL_RANK):
        if opt.model == 'list':
            m = hub.list(repo1) + hub.list(repo2)  # models
            LOGGER.info('\nAvailable models. Usage: python classifier.py --model MODEL\n' + '\n'.join(m))
            return
        elif opt.model.startswith('yolov5'):  # YOLOv5 models, i.e. yolov5s, yolov5m
            from models.yolo import ClassificationModel
            model = smart_hub_load(repo1,
                                   opt.model,
                                   pretrained=pretrained,
                                   _verbose=False,
                                   autoshape=False,
                                   device='cpu')  # detection model
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # classification model
        elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            model = torchvision.models.__dict__[opt.model](weights='IMAGENET1K_V1' if pretrained else None)
            update_classifier_model(model, nc)  # update class count
        else:
            m = hub.list(repo1) + hub.list(repo2)  # models
            raise ModuleNotFoundError(f'--model {opt.model} not found. Available models are: \n' + '\n'.join(m))
    for p in model.parameters():
        p.requires_grad = True  # for training
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    model = model.to(device)
    model.names = names  # attach class names

    # Info
    if RANK in {-1, 0}:
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow(denormalize(images[:25]), labels[:25], names=names, f=save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)  # log model

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=5e-5)

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Train
    t0 = time.time()
    criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    val = test_dir.stem  # 'val' or 'test'
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = test(ema.ema, testloader, names, criterion, pbar=pbar)  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy

        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(f'\nTraining complete {(time.time() - t0) / 3600:.3f} hours.'
                    f"\nResults saved to {colorstr('bold', save_dir)}")

        # Show predictions
        images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        file = imshow(denormalize(images), labels, pred, names, verbose=True, f=save_dir / 'test_images.jpg')
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_images(file, name='Test Examples (true-predicted)', epoch=epoch)
        logger.log_model(best, epochs, metadata=meta)


@smart_inference_mode()
def test(model, dataloader, names, criterion=None, verbose=False, pbar=None):
    model.eval()
    device = next(model.parameters()).device
    pred, targets, loss = [], [], 0
    n = len(dataloader)  # number of batches
    action = 'validating' if dataloader.dataset.root.stem == 'val' else 'testing'
    desc = f"{pbar.desc[:-36]}{action:>36}"
    bar = tqdm(dataloader, desc, n, False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0)
    with amp.autocast(enabled=device.type != 'cpu'):
        for images, labels in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            y = model(images)
            pred.append(y.argsort(1, descending=True)[:, :5])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  # all classes
        LOGGER.info(f"{'Class':>20}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>20}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in enumerate(names):
            aci = acc[targets == i]
            top1i, top5i = aci.mean(0).tolist()
            LOGGER.info(f"{c:>20}{aci.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

    return top1, top5, loss


@smart_inference_mode()
def classify(model, size=128, file='../datasets/mnist/test/3/30.png', plot=False):
    # YOLOv5 classification model inference
    import cv2
    import numpy as np
    import torch.nn.functional as F

    resize = torch.nn.Upsample(size=(size, size), mode='bilinear', align_corners=False)  # image resize

    # Image
    im = cv2.imread(str(file))[..., ::-1]  # HWC, BGR to RGB
    im = np.ascontiguousarray(np.asarray(im).transpose((2, 0, 1)))  # HWC to CHW
    im = torch.tensor(im).float().unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    im = resize(im)

    # Inference
    results = model(normalize(im))
    p = F.softmax(results, dim=1)  # probabilities
    i = p.argmax()  # max index
    LOGGER.info(f'{file} prediction: {i} ({p[0, i]:.2f})')

    # Plot
    if plot:
        imshow(im, f=Path(file).name)

    return p


def imshow(img, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path('images.jpg')):
    # Show classification image grid with labels (optional) and predictions (optional)
    import matplotlib.pyplot as plt

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(img.cpu(), len(img), dim=0)  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n ** 0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s, fontsize=8, verticalalignment='top')
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(colorstr('imshow: ') + f"examples saved to {f}")
    if verbose:
        if labels is not None:
            LOGGER.info('True:     ' + ' '.join(f'{names[i]:3s}' for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info('Predicted:' + ' '.join(f'{names[i]:3s}' for i in pred[:nmax]))
    return f


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
    parser.add_argument('--data', type=str, default='mnist', help='cifar10, cifar100, mnist or fashion-mnist')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


def run(**kwargs):
    # Usage: import classifier; classifier.run(data=mnist, imgsz=320, model='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
