# Resume all interrupted trainings in yolov5/ dir including DDP trainings
# Usage: $ python utils/aws/resume.py

import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

port = 0  # --master_port
path = Path('').resolve()
for last in path.rglob('*/**/last.pt'):
    ckpt = torch.load(last)
    if ckpt['optimizer'] is None:
        continue

    # Load opt.yaml
    with open(last.parent.parent / 'opt.yaml') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    # Get device count
    d = opt['device'].split(',')  # devices
    nd = len(d)  # number of devices
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)  # distributed data parallel

    if ddp:  # multi-GPU
        port += 1
        cmd = f'python -m torch.distributed.launch --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # single-GPU
        cmd = f'python train.py --resume {last}'

    cmd += ' > /dev/null 2>&1 &'  # redirect output to dev/null and run in daemon thread
    print(cmd)
    os.system(cmd)
