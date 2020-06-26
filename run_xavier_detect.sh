#!/bin/bash

sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --mount src=$(pwd)/weights,target=/usr/src/app/weights/,type=bind \
  --mount src=$(pwd)/inference,target=/usr/src/app/inference/,type=bind \
  yolov5-app:latest python3 detect.py
