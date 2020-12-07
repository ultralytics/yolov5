import argparse
import logging
import os
from pathlib import Path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.cuda import amp
from tqdm import tqdm

from models.common import Classify
from utils.general import set_logging, check_file, increment_path
from utils.torch_utils import model_info, select_device

# Settings
logger = logging.getLogger(__name__)
set_logging()


# Show images
def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0)))  # unnormalize
    plt.savefig('images.jpg')


def train():
    data, bs, epochs, nw = opt.data, opt.batch_size, opt.epochs, opt.workers

    # Download Dataset
    if not Path(f'../{data}').is_dir():
        f = 'tmp.zip'
        torch.hub.download_url_to_file(f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip', f)
        os.system(f'unzip -q {f} -d ../ && rm {f}')  # unzip

    # Transforms
    trainform = T.Compose([T.RandomGrayscale(p=0.01),
                           T.RandomHorizontalFlip(p=0.5),
                           T.RandomAffine(degrees=1, translate=(.2, .2), scale=(1 / 1.5, 1.5),
                                          shear=(-1, 1, -1, 1), fillcolor=(114, 114, 114)),
                           # T.Resize([128, 128]),
                           T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])  # PILImage from [0, 1] to [-1, 1]
    testform = T.Compose(trainform.transforms[-2:])

    # Dataloaders
    trainset = torchvision.datasets.ImageFolder(root=f'../{data}/train', transform=trainform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw)
    testset = torchvision.datasets.ImageFolder(root=f'../{data}/test', transform=testform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=nw)
    names = trainset.classes
    nc = len(names)
    print(f'Training {opt.model} on {data} dataset with {nc} classes...')

    # Show images
    # images, labels = iter(trainloader).next()
    # imshow(torchvision.utils.make_grid(images[:16]))
    # print(' '.join('%5s' % names[labels[j]] for j in range(16)))

    # Model
    if opt.model.startswith('yolov5'):
        # YOLOv5 Classifier
        model = torch.hub.load('ultralytics/yolov5', opt.model, pretrained=True)
        model.model = model.model[:8]
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else sum([x.in_channels for x in m.m])  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
    elif opt.model in torch.hub.list('rwightman/gen-efficientnet-pytorch'):
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', opt.model, pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, nc)
    else:  # try torchvision
        model = torchvision.models.__dict__[opt.model](pretrained=True)
        model.fc = nn.Linear(model.fc.weight.shape[1], nc)

    # print(model)

    model_info(model)

    # Optimizer
    lr0 = 0.0001 * bs  # intial lr
    lrf = 0.01  # final lr (fraction of lr0)
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

    # Scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Train
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    # scaler = amp.GradScaler(enabled=cuda)
    print(f"\n{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        mloss = 0.  # mean loss
        model.train()
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))  # progress bar
        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)
            images = F.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                loss = criterion(model(images), labels)

            # Backward
            loss.backward()  # scaler.scale(loss).backward()

            # Optimize
            optimizer.step()  # scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

            # Print
            mloss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}"

            # Test
            if i == len(pbar) - 1:
                test(model, testloader, names, criterion, pbar=pbar)  # test

        # Test
        scheduler.step()

    # Show predictions
    # images, labels = iter(testloader).next()
    # predicted = torch.max(model(images), 1)[1]
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % names[labels[j]] for j in range(4)))
    # print('Predicted: ', ' '.join('%5s' % names[predicted[j]] for j in range(4)))


def test(model, dataloader, names, criterion=None, verbose=False, pbar=None):
    model.eval()
    pred, targets, loss = [], [], 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = F.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)
            y = model(images)
            pred.append(torch.max(y, 1)[1])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets == pred).float()

    if pbar:
        pbar.desc += f"{loss / len(dataloader):<12.3g}{correct.mean().item():<12.3g}"

    if verbose:  # all classes
        print('%10s' * 3 % ('class', 'number', 'accuracy'))
        print('%10s%10s%10.5g' % ('all', correct.shape[0], correct.mean().item()))
        for i, c in enumerate(names):
            t = correct[targets == i]
            print('%10s%10s%10.5g' % (c, t.shape[0], t.mean().item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
    parser.add_argument('--data', type=str, default='cifar100', help='cifar10, cifar100 or mnist')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[128, 128], help='[train, test] image sizes')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != 'cpu'
    opt.hyp = check_file(opt.hyp)  # check files
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 if 1
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    train()
