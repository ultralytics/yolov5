#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download ImageNet10 dataset https://image-net.org (first 10 classes of ILSVRC2012 train/val)
# Example usage: bash data/scripts/get_imagenet10.sh
# parent
# ├── yolov5
# └── datasets
#     └── imagenet10  ← downloads here

# Make dir
d='../datasets/imagenet10' # unzip directory
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet10.zip
unzip imagenet10.zip && rm imagenet10.zip
