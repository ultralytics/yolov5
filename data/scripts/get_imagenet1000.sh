#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download ILSVRC2012 ImageNet dataset https://image-net.org
# Example usage: bash data/scripts/get_imagenet1000.sh
# parent
# ├── yolov5
# └── datasets
#     └── imagenet1000  ← downloads here

# Make dir
d='../datasets/imagenet1000' # unzip directory
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet1000.zip
unzip imagenet1000.zip && rm imagenet1000.zip
