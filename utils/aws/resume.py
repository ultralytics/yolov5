# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Resume all interrupted trainings in yolov5/ dir including DDP trainings
# Usage: $ python utils/aws/resume.py

import subprocess
import sys
from pathlib import Path

import torch
import yaml
from ultralytics.utils.patches import torch_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

port = 0  # --master_port
path = Path("").resolve()
for last in path.rglob("*/**/last.pt"):
    ckpt = torch_load(last)
    if ckpt["optimizer"] is None:
        continue

    # Load opt.yaml
    with open(last.parent.parent / "opt.yaml", errors="ignore") as f:
        opt = yaml.safe_load(f)

    # Get device count
    d = opt["device"].split(",")  # devices
    nd = len(d)  # number of devices
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)  # distributed data parallel

    if ddp:  # multi-GPU
        port += 1
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nd),
            "--master_port",
            str(port),
            "train.py",
            "--resume",
            str(last),
        ]
    else:  # single-GPU
        cmd = ["python", "train.py", "--resume", str(last)]

    print(" ".join(cmd))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # run in daemon thread
