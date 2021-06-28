#!/bin/sh
nohup python3 detect.py \
    --weights /usr/src/weights/best.pt \
    --img 640 \
    --conf-thres 0.25 \
    --source /usr/src/data \
    --save-txt