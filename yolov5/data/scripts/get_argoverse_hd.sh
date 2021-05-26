#!/bin/bash
# Argoverse-HD dataset (ring-front-center camera) http://www.cs.cmu.edu/~mengtial/proj/streaming/
# Download command: bash data/scripts/get_argoverse_hd.sh
# Train command: python train.py --data argoverse_hd.yaml
# Default dataset location is next to YOLOv5:
#   /parent_folder
#     /argoverse
#     /yolov5

# Download/unzip images
d='../argoverse/' # unzip directory
mkdir $d
url=https://argoverse-hd.s3.us-east-2.amazonaws.com/
f=Argoverse-HD-Full.zip
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f &# download, unzip, remove in background
wait                                              # finish background tasks

cd ../argoverse/Argoverse-1.1/
ln -s tracking images

cd ../Argoverse-HD/annotations/

python3 - "$@" <<END
import json
from pathlib import Path

annotation_files = ["train.json", "val.json"]
print("Converting annotations to YOLOv5 format...")

for val in annotation_files:
    a = json.load(open(val, "rb"))

    label_dict = {}
    for annot in a['annotations']:
        img_id = annot['image_id']
        img_name = a['images'][img_id]['name']
        img_label_name = img_name[:-3] + "txt"

        cls = annot['category_id']  # instance class id
        x_center, y_center, width, height = annot['bbox']
        x_center = (x_center + width / 2) / 1920.  # offset and scale
        y_center = (y_center + height / 2) / 1200.  # offset and scale
        width /= 1920.  # scale
        height /= 1200.  # scale

        img_dir = "./labels/" + a['seq_dirs'][a['images'][annot['image_id']]['sid']]

        Path(img_dir).mkdir(parents=True, exist_ok=True)
        if img_dir + "/" + img_label_name not in label_dict:
            label_dict[img_dir + "/" + img_label_name] = []

        label_dict[img_dir + "/" + img_label_name].append(f"{cls} {x_center} {y_center} {width} {height}\n")

    for filename in label_dict:
        with open(filename, "w") as file:
            for string in label_dict[filename]:
                file.write(string)

END

mv ./labels ../../Argoverse-1.1/
