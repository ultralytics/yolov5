#!/bin/bash
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128
# Download command: bash data/scripts/get_coco128.sh
# Train command: python train.py --data coco128.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /coco128
#     /yolov5

# Download/unzip images and labels
d='../' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco128.zip' # or 'coco2017labels-segments.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

wait # finish background tasks
