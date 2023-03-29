#!/bin/bash
# This script can be periodically executed to build the latest voxel yolo image
# and push it to ECR
set -euo pipefail

date_utc=$(date --utc +'%Y-%m-%dT%H-%M-%S')
voxel_yolo_image=203670452561.dkr.ecr.us-west-2.amazonaws.com/sematic:voxel-yolov5-$date_utc
docker build -t "$voxel_yolo_image" -f utils/docker/Dockerfile .
docker push "$voxel_yolo_image"
echo "Please update yolov5.Dockerfile in voxel repo with following url to pull correct image"
echo "FROM $voxel_yolo_image"