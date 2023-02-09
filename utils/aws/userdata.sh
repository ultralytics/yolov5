#!/bin/bash
# AWS EC2 instance startup script https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html
# This script will run only once on first instance start (for a re-start script see mime.sh)
# /home/ubuntu (ubuntu) or /home/ec2-user (amazon-linux) is working dir
# Use >300 GB SSD

cd home/ubuntu
if [ ! -d yolov5 ]; then
  echo "Running first-time script." # install dependencies, download COCO, pull Docker
  git clone https://github.com/ultralytics/yolov5 -b master && sudo chmod -R 777 yolov5
  cd yolov5
  bash data/scripts/get_coco.sh && echo "COCO done." &
  sudo docker pull ultralytics/yolov5:latest && echo "Docker done." &
  python -m pip install --upgrade pip && pip install -r requirements.txt && python detect.py && echo "Requirements done." &
  wait && echo "All tasks done." # finish background tasks
else
  echo "Running re-start script." # resume interrupted runs
  i=0
  list=$(sudo docker ps -qa) # container list i.e. $'one\ntwo\nthree\nfour'
  while IFS= read -r id; do
    ((i++))
    echo "restarting container $i: $id"
    sudo docker start $id
    # sudo docker exec -it $id python train.py --resume # single-GPU
    sudo docker exec -d $id python utils/aws/resume.py # multi-scenario
  done <<<"$list"
fi
