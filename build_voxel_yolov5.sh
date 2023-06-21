#!/bin/bash
# This script can be periodically executed to build the latest voxel yolo image
# and push it to ECR
set -euo pipefail

date_utc=$(date --utc +'%Y-%m-%dT%H-%M-%S')
voxel_yolo_image=203670452561.dkr.ecr.us-west-2.amazonaws.com/yolov5:voxel-yolov5-$date_utc
docker build -t "$voxel_yolo_image" -f utils/docker/Dockerfile .
docker push "$voxel_yolo_image"
dockerSHA=$(docker inspect --format='{{index .RepoDigests 0}}' "$voxel_yolo_image" | perl -wnE'say /sha256.*/g')
echo "$dockerSHA"
echo "Please update Voxel Repo WORKSPACE with following yolov5_repush info"
echo 'authenticated_container_pull(
    name = "yolov5_repush",
    digest = "'"${dockerSHA}"'",
    registry = "203670452561.dkr.ecr.us-west-2.amazonaws.com",
    repository = "yolov5",
    tag = "voxel-yolov5-'"${date_utc}"'",
)'