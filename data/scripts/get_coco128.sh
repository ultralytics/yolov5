#!/bin/bash
# Copyright Ultralytics https://ultralytics.com, licensed under GNU GPL v3.0
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# YOLOv5 🚀 example usage: bash data/scripts/get_coco128.sh
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here

# Download/unzip images and labels
d='../datasets' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco128.zip' # or 'coco2017labels-segments.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

wait # finish background tasks
