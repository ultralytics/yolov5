#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download ImageNet100 dataset https://image-net.org (first 100 classes of ILSVRC2012 train/val)
# Example usage: bash data/scripts/get_imagenet100.sh
# parent
# ├── yolov5
# └── datasets
#     └── imagenet100  ← downloads here

# Make dir
d='../datasets/imagenet100' # unzip directory
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet100.zip
unzip imagenet100.zip && rm imagenet100.zip
