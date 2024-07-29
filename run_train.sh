#!/bin/bash


python train.py \
       --weights yolov5n.pt \
       --cfg models/yolov5n-lsknet.yaml \
       --data data/learning.yaml \
       --batch-size 2 \
       --imgsz 640 \
       --name learning
