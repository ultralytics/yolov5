#!/bin/bash

echo "Running first-time script 64, 8."
python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt \
--hyp data/hyp.qat.yaml --calib-batch-size 64 --num-calib-batch 8 --device=1


echo "Running first-time script 64, 16."
python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt \
--hyp data/hyp.qat.yaml --calib-batch-size 64 --num-calib-batch 16 --device=1


echo "Running first-time script 32, 16."
python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt \
--hyp data/hyp.qat.yaml --calib-batch-size 32 --num-calib-batch 16 --device=1


echo "Running first-time script 128, 8."
python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt \
--hyp data/hyp.qat.yaml --calib-batch-size 128 --num-calib-batch 8 --device=1


echo "Running first-time script 64, 24."
python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt \
--hyp data/hyp.qat.yaml --calib-batch-size 64 --num-calib-batch 24 --device=1