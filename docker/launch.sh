#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"0"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm --name yolov5_quant -p 80:8888 \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=16g \
  -v $(dirname $(pwd)):/root/space/projects \
  yolov5_quant $CMD