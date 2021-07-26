#!/bin/bash
# Copyright Ultralytics https://ultralytics.com, licensed under GNU GPL v3.0
# Download latest models from https://github.com/ultralytics/yolov5/releases
# YOLOv5 🚀 example usage: bash path/to/download_weights.sh
# parent
# └── yolov5
#     ├── yolov5s.pt  ← downloads here
#     ├── yolov5m.pt
#     └── ...

python - <<EOF
from utils.google_utils import attempt_download

for x in ['s', 'm', 'l', 'x']:
    attempt_download(f'yolov5{x}.pt')

EOF
