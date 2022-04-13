#!/bin/sh
echo "folder $1"
# python3 detect.py \
#     --weights /usr/src/weights/singletext.pt /usr/src/weights/graphicalwithtext.pt \
#     --img 640 \
#     --conf-thres 0.25 \
#     --augment\
#     --source "/usr/src/data/$1" \
#     --save-txt\
#     --name $1\
#     --device 0

python3 detect.py \
    --weights /usr/src/weights/$2.pt\
    --img 1280 \
    --conf-thres 0.25 \
    --augment\
    --source "/usr/src/data/$1" \
    --save-txt\
    --name $1\
    --device 0