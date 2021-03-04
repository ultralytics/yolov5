#!/bin/bash
# Argoverse dataset (ring-front-center camera) https://www.argoverse.org/data.html
# Download command: bash data/scripts/get_argoverse.sh
# Train command: python train.py --data argoverse.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /argoverse
#     /yolov5

# Download/unzip images
p='../argoverse/'
d='../argoverse/Argoverse-1.1/' # unzip directory
url=https://s3.amazonaws.com/argoai-argoverse/
f1='tracking_train1_v1.1.tar.gz' # 43G
f2='tracking_train2_v1.1.tar.gz' # 52G
f3='tracking_train3_v1.1.tar.gz' # 46G
f4='tracking_train4_v1.1.tar.gz' # 9.8G
f2='tracking_val_v1.1.tar.gz'    # 57G
f3='tracking_test_v1.1.tar.gz'   # 49G (optional)
mkdir $p
mkdir $d
for f in $f1 $f2 $f3 $f4 $f5 $f6; do
  echo 'Downloading' $url$f '...'
  wget $url$f -O $f && tar -xvzf $f -C $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks

old_dir='../argoverse/Argoverse-1.1/argoverse-tracking/'
new_dir='../argoverse/Argoverse-1.1/images/'
mv $old_dir $new_dir

# move all train files to single folder
mkdir '../argoverse/Argoverse-1.1/images/train'
mv ../argoverse/Argoverse-1.1/images/train1/* '../argoverse/Argoverse-1.1/images/train/' && rm -r ../argoverse/Argoverse-1.1/images/train1/
mv ../argoverse/Argoverse-1.1/images/train2/* '../argoverse/Argoverse-1.1/images/train/' && rm -r ../argoverse/Argoverse-1.1/images/train2/
mv ../argoverse/Argoverse-1.1/images/train3/* '../argoverse/Argoverse-1.1/images/train/' && rm -r ../argoverse/Argoverse-1.1/images/train3/
mv ../argoverse/Argoverse-1.1/images/train4/* '../argoverse/Argoverse-1.1/images/train/' && rm -r ../argoverse/Argoverse-1.1/images/train4/

# get all images from ring_front_center camera and delete the rest
for split in "train" "test" "val"; do 
  for d in ../argoverse/Argoverse-1.1/images/$split/*; do
      if [ -d "$d" ]; then
          for f in $d/ring_front_center/*; do
              mv $f ../argoverse/Argoverse-1.1/images/$split/ && rm -rf $d
          done
      fi
  done
done

# Download labels
labels_url = 'https://github.com/karthiksharma98/sap-starterkit/releases/download/yolov5-labels/'
labels_file = 'labels.zip'
mkdir '../argoverse/Argoverse-1.1/labels'
wget $labels_url -O $labels_file && unzip $labels_file -d ../argoverse/Argoverse-1.1/labels/ && rm $labels_file 



