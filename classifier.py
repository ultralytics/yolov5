# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - train:
    $ python classifier.py --model yolov5s --data mnist --epochs 5 --img 128
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
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import Classify, DetectMultiBackend
from utils.augmentations import denormalize, normalize
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, check_file, check_git_status, check_requirements, check_version, colorstr, download,
                           increment_path, init_seeds, print_args)
from utils.loggers import GenericLogger
from utils.torch_utils import de_parallel, model_info, ModelEMA, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train():
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        Path(opt.save_dir), opt.data, opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), opt.imgsz, \
        not opt.from_scratch
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Download Dataset
    data_dir = FILE.parents[1] / 'datasets' / data
    with torch_distributed_zero_first(LOCAL_RANK):
        if not data_dir.is_dir():
            url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
            download(url, dir=data_dir.parent)

    # Dataloaders
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=opt.cache,
                                                   rank=LOCAL_RANK,
                                                   workers=nw)

    if RANK in {-1, 0}:
        test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=opt.cache,
                                                      rank=-1,
                                                      workers=nw)

    # Initialize
    names = trainloader.dataset.classes  # class names
    nc = len(names)  # number of classes
    LOGGER.info(f'Training {opt.model} on {data} dataset with {nc} classes...')
    init_seeds(1 + RANK)

    # Show images
    images, labels = iter(trainloader).next()
    imshow(denormalize(images[:64]), labels[:64], names=names, f=save_dir / 'train_images.jpg')

    # Model
    repo1, repo2 = 'ultralytics/yolov5', 'rwightman/gen-efficientnet-pytorch'
    with torch_distributed_zero_first(LOCAL_RANK):
        if opt.model.startswith('yolov5'):  # YOLOv5 Classifier
            try:
                model = hub.load(repo1, opt.model, pretrained=pretrained, autoshape=False)
            except Exception:
                model = hub.load(repo1, opt.model, pretrained=pretrained, autoshape=False, force_reload=True)
            if isinstance(model, DetectMultiBackend):
                model = model.model  # unwrap DetectMultiBackend
            model.model = model.model[:10] if opt.model.endswith('6') else model.model[:8]  # backbone
            m = model.model[-1]  # last layer
            ch = m.conv.in_channels if hasattr(m, 'conv') else sum(x.in_channels for x in m.m)  # ch into module
            c = Classify(ch, nc)  # Classify()
            c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
            model.model[-1] = c  # replace
            for p in model.parameters():
                p.requires_grad = True  # for training
        elif opt.model in hub.list(repo2):  # i.e. efficientnet_b0
            model = hub.load(repo2, opt.model, pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, nc)
        else:  # try torchvision
            model = torchvision.models.__dict__[opt.model](pretrained=pretrained)
            model.fc = nn.Linear(model.fc.weight.shape[1], nc)
    model = model.to(device)
    if RANK in {-1, 0}:
        model_info(model)  # print(model)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Optimizer
    lr0 = 0.01 * (1 if opt.optimizer.startswith('Adam') else 0.01 * bs)  # initial lr
    lrf = 0.001  # final lr (fraction of lr0)
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr0 / 10)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

    # Scheduler
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # DDP mode
    if cuda and RANK != -1:
        if check_version(torch.__version__, '1.11.0'):
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
        else:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Train
    t0 = time.time()
    criterion = nn.CrossEntropyLoss()  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...\n\n'
                f"{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':10s}{mem:10s}{tloss:<12.3g}"

                # Test
                if i == len(pbar) - 1:  # last batch
                    fitness, vloss = test(ema.ema, testloader, names, criterion, pbar=pbar)  # test accuracy, loss

        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            lr = optimizer.param_groups[0]['lr']  # learning rate
            logger.log_metrics({"train/loss": tloss, "val/loss": vloss, "metrics/accuracy": fitness, "lr/0": lr}, epoch)

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
        images, labels = (x[:64] for x in iter(testloader).next())  # first 30 images and labels
        images = images.to(device)
        pred = torch.max(model(images), 1)[1]
        imshow(denormalize(images), labels, pred, names, verbose=True, f=save_dir / 'test_images.jpg')


@torch.no_grad()
def test(model, dataloader, names, criterion=None, verbose=False, pbar=None):
    model.eval()
    pred, targets, loss = [], [], 0
    n = len(dataloader)  # number of batches
    desc = f'{pbar.desc}validating'
    bar = tqdm(dataloader, desc, n, False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        y = model(images)
        pred.append(torch.max(y, 1)[1])
        targets.append(labels)
        if criterion:
            loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets == pred).float()

    if pbar:
        pbar.desc += f"{loss:<12.3g}{correct.mean().item():<12.3g}"

    accuracy = correct.mean().item()
    if verbose:  # all classes
        LOGGER.info(f"{'class':10s}{'number':10s}{'accuracy':10s}")
        LOGGER.info(f"{'all':10s}{correct.shape[0]:10s}{accuracy:10.5g}")
        for i, c in enumerate(names):
            t = correct[targets == i]
            LOGGER.info(f"{c:10s}{t.shape[0]:10s}{t.mean().item():10.5g}")

    return accuracy, loss


@torch.no_grad()
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


def imshow(img, labels=None, pred=None, names=None, nmax=64, verbose=False, f=Path('images.jpg')):
    # Show classification image grid with labels (optional) and predictions (optional)
    import matplotlib.pyplot as plt

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(img.cpu(), len(img), dim=0)  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n ** 0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m, tight_layout=True)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s)

    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info(colorstr('imshow: ') + f"examples saved to {f}")

    if verbose and labels is not None:
        LOGGER.info('True:     ' + ' '.join(f'{names[i]:3s}' for i in labels))
    if verbose and pred is not None:
        LOGGER.info('Predicted:' + ' '.join(f'{names[i]:3s}' for i in pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
    parser.add_argument('--data', type=str, default='mnist', help='cifar10, cifar100, mnist or mnist-fashion')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=128, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--from-scratch', '--scratch', action='store_true', help='train model from scratch')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()

    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # Parameters
    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != 'cpu'
    opt.hyp = check_file(opt.hyp)  # check files
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    # TODO: Remove resize as redundant with augmentations
    resize = torch.nn.Upsample(size=(opt.imgsz, opt.imgsz), mode='bilinear', align_corners=False)  # image resize

    # DDP
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        # TODO assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    train()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()
