#!/bin/bash
# Hyperparameter evolution commands (avoids CUDA memory leakage issues)
# Replaces train.py python generations 'for' loop with a bash 'for' loop

# Start on 4-GPU machine
#for i in 0 1 2 3; do
#  t=ultralytics/yolov5:evolve && sudo docker pull $t && sudo docker run -d --ipc=host --gpus all -v "$(pwd)"/VOC:/usr/src/VOC $t bash utils/evolve.sh $i
#  sleep 60 # avoid simultaneous evolve.txt read/write
#done

# Hyperparameter evolution commands
while true; do
  # python train.py --batch 64 --weights yolov5m.pt --data voc.yaml --img 512 --epochs 50 --evolve --bucket ult/evolve/voc --device $1
  python train.py --batch 40 --weights yolov5m.pt --data coco.yaml --img 640 --epochs 30 --evolve --bucket ult/evolve/coco --device $1
done
