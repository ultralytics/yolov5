#!/bin/bash
# Download latest model from https://github.com/ultralytics/yolov5/releases

python - <<EOF
from utils.google_utils import attempt_download

for x in ['s', 'm', 'l', 'x']:
    attempt_download(f'yolov5{x}.pt')

EOF
