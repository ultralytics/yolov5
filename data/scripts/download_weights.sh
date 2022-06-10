#!/bin/bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Example usage: bash path/to/download_weights.sh
# parent
# └── yolov5
#     ├── yolov5s.pt  ← downloads here
#     ├── yolov5m.pt
#     └── ...

python - <<EOF
from utils.downloads import attempt_download

models = ['n', 's', 'm', 'l', 'x']
models.extend([x + '6' for x in models])  # add P6 models

for x in models:
    attempt_download(f'yolov5{x}.pt')

EOF
