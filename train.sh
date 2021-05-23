#!/usr/bin/env bash
# 注意不同的任务需要修改 # runs/exp train.py

# 工作服(反光衣) 2类 单独训练
# docker run -it --gpus all -v /home/epi/gary/:/home/epi/gary nvcr.io/nvidia/pytorch:21.02-py3
# cd root@dc7f301c3e74:/home/epi/gary/ml_work_space/gary_orig/github/yolov5#
python train.py --img 640 --batch 8 --epochs 100 --data ./data/Reflective_vests.yaml --cfg ./models/yolov5s_fs.yaml --weights './yolov5s.pt' --device 0
