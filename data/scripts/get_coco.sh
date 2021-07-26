#!/bin/bash
# Copyright Ultralytics https://ultralytics.com, licensed under GNU GPL v3.0
# Download COCO 2017 dataset http://cocodataset.org
# YOLOv5 🚀 example usage: bash data/scripts/get_coco.sh
# parent
# ├── yolov5
# └── datasets
#     └── coco  ← downloads here

# Download/unzip labels
d='../datasets' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels.zip' # or 'coco2017labels-segments.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

# Download/unzip images
d='../datasets/coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
f2='val2017.zip'   # 1G, 5k images
f3='test2017.zip'  # 7G, 41k images (optional)
for f in $f1 $f2; do
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks
