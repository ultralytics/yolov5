#!/bin/bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Download ILSVRC2012 ImageNet dataset https://image-net.org
# Example usage: bash data/scripts/get_imagenet.sh
# parent
# ├── yolov5
# └── datasets
#     └── imagenet  ← downloads here

# Download
d='../datasets/imagenet' # unzip directory
mkdir -p $d && cd $d
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar   # 6.3G, 50000 images
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar # 138G, 1281167 images

# Extract train
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME; do
  mkdir -p "${NAME%.tar}"
  tar -xvf "${NAME}" -C "${NAME%.tar}"
  rm -f "${NAME}"
done
cd ..

# Extract val and move images into subdirectories
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# Delete corrupted image (optional: PNG under JPEG name that may cause dataloaders to fail)
# rm train/n04266014/n04266014_10835.JPEG

# TFRecords (optional)
# wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
