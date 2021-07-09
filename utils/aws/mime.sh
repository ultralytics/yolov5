# AWS EC2 instance startup 'MIME' script https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/
# This script will run on every instance restart, not only on first start
# --- DO NOT COPY ABOVE COMMENTS WHEN PASTING INTO USERDATA ---

Content-Type: multipart/mixed; boundary="//"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="cloud-config.txt"

#cloud-config
cloud_final_modules:
- [scripts-user, always]

--//
Content-Type: text/x-shellscript; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="userdata.txt"

#!/bin/bash
cd home/ubuntu
if [ ! -d yolov5 ]; then
  echo "Running first-time script." # install dependencies, download COCO, pull Docker
  git clone https://github.com/ultralytics/yolov5 -b exp/albumentations && sudo chmod -R 777 yolov5
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
--//
