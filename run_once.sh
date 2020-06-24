sudo docker run --gpus all --ipc=host -it \
  -v "$(pwd)"/coco:/usr/src/coco \
  -v "$(pwd)"/inference:/usr/src/app/inference \
  ultralytics/yolov5:latest

