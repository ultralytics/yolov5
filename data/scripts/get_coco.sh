#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash data/scripts/get_coco.sh
# Train command: python train.py --data coco.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /coco
#     /yolov5

# Download/unzip labels
echo 'Downloading COCO 2017 labels ...'
d='../' # unzip directory
f='coco2017labels.zip' && curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/$f -o $f
unzip -q $f -d $d && rm $f

# Download/unzip images
echo 'Downloading COCO 2017 images ...'
d='../coco/images' # unzip directory
f='train2017.zip' && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f -d $d && rm $f # 19G, 118k images
f='val2017.zip' && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f -d $d && rm $f   # 1G, 5k images
# f='test2017.zip' && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f -d $d && rm $f  # 7G,  41k images
