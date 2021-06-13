#!/bin/sh
for i in 0 1 2 3 4 5 6 7; do
    nohup python train.py \
          --cache \
          --img-size 640 \
          --batch 16 \
          --epochs 50 \
          --data /usr/src/data/mh-newspapers/newspapers.yaml \
          --weights /usr/src/data/backbone-training-800-best.pt \
          --single-cls \
          --adam \
          --workers 8 \
          --original-image-path /usr/src/data/mh-newspapers/ \
          --temp-image-path /usr/src/data/temp/ \
          --name output_4k-final-singletext \
          --project YOLOv5-mh-data-evolve-all-text-image-manipulation \
          --evolve --device $i > evolve_gpu_$i.log &
done