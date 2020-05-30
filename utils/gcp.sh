#!/usr/bin/env bash

# New VM
if [ ! -d ./coco ]
then
  echo "COCO folder not found. Running startup script."
  git clone https://github.com/ultralytics/yolov5
  # git clone -b test --depth 1 https://github.com/ultralytics/yolov5 test  # branch
  # sudo apt-get install zip
  # git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
  # sudo conda install -yc conda-forge scikit-image pycocotools
  python3 -c "from yolov5.utils.google_utils import gdrive_download; gdrive_download('1rrL-Jbc68iHiGjXOYc8u9tKfFiOX21Tn','coco2017.zip')"
  python3 -c "from yolov5.utils.google_utils import gdrive_download; gdrive_download('1Y6Kou6kEB0ZEMCCpJSKStCor4KAReE43','coco2017.zip')"
  sudo docker pull ultralytics/coco:198

  # Add 64GB swap
  sudo fallocate -l 64G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  free -h  # check memory

  # sudo reboot now
fi
n=198 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp2.yaml --bucket ult/coco --name $n --data data/coco.yaml


# Evolve coco
sudo -s
t=ultralytics/yolov3:evolve
# docker kill $(docker ps -a -q --filter ancestor=$t)
for i in 0 1 6 7
do
  docker pull $t && docker run --gpus all -d --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash utils/evolve.sh $i
  sleep 30
done


# kmean_anchors(path='../coco/train2017.txt', n=12, img_size=(256, 1024), thr=0.10, gen=10000)
# 0.10 iou_thr: 1.000 best possible recall, 6.15 anchors > thr
# n=12, img_size=(256, 1024), IoU_all=0.188/0.641-mean/best, IoU>thr=0.338-mean: 7,9,  13,17,  30,21,  18,38,  31,63,  54,38,  52,110,  90,69,  98,187,  164,116,  218,255,  448,414

# from yolov4l_10iou
# computed with utils.kmean_anchors(path='../coco/train2017.txt', n=12, img_size=(320, 1024), thr=0.10, gen=1000)
# Evolving anchors: 100%|█████████████████| 1000/1000 [39:57<00:00,  2.40s/it]
# 0.10 iou_thr: 0.998 best possible recall, 6.07 anchors > thr
# n=12, img_size=(320, 1024), IoU_all=0.187/0.635-mean/best, IoU>thr=0.339-mean: 9,13,  20,21,  21,50,  43,34,  44,89,  84,59,  76,164,  133,105,  216,176,  153,267,  312,331,  623,467

# from yolov4l_10iou_9anchors
# computed with utils.kmean_anchors(path='../coco/train2017.txt', n=9, img_size=(320, 1024), thr=0.10, gen=1000)
# Evolving anchors: 100%|█████████████████| 1000/1000 [31:26<00:00,  1.89s/it]
# 0.10 iou_thr: 0.998 best possible recall, 4.60 anchors > thr
# n=9, img_size=(320, 1024), IoU_all=0.190/0.604-mean/best, IoU>thr=0.342-mean: 9,13,  20,26,  29,58,  55,37,  57,115,  105,77,  133,165,  254,280,  508,450

# ar < 5
#Evolving anchors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [14:56<00:00,  1.00it/s]
#0.10 iou_thr: 0.999 best possible recall, 4.63 anchors > thr
#n=9, img_size=(320, 1024), IoU_all=0.187/0.601-mean/best, IoU>thr=0.334-mean: 9,11,  15,35,  36,21,  34,61,  86,55,  66,145,  164,110,  185,262,  450,369

#Evolving anchors:  10%|████████████                                                                                                           | 1015/10000 [15:59<1:56:24,  1.29it/s]
#5.00 iou_thr: 1.000 best possible recall, 4.79 anchors > thr
#n=9, img_size=(320, 1024), IoU_all=9.193/1.535-mean/best, IoU>thr=2.745-mean: 8,8,  14,24,  41,21,  24,56,  64,57,  60,140,  154,102,  174,262,  443,340

#Evolving anchors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [2:15:31<00:00,  1.25it/s]
#5.00 iou_thr: 1.000 best possible recall, 4.80 anchors > thr
#n=9, img_size=(320, 1024), IoU_all=9.106/1.531-mean/best, IoU>thr=2.744-mean: 8,8,  12,24,  33,19,  25,53,  64,48,  58,128,  145,93,  164,240,  419,349


Evolving anchors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [21:39<00:00,  7.83it/s]
0.20 iou_thr: 0.992 best possible recall, 3.44 anchors > thr
n=9, img_size=(640, 640), IoU_all=0.199/0.614-mean/best, IoU>thr=0.421-mean: 9,11,  16,25,  38,25,  26,54,  70,50,  51,102,  113,109,  161,218,  366,343

Evolving anchors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [16:34<00:00, 10.05it/s]
0.10 iou_thr: 0.999 best possible recall, 4.87 anchors > thr
n=9, img_size=(640, 640), IoU_all=0.200/0.614-mean/best, IoU>thr=0.342-mean: 10,12,  16,27,  37,25,  27,56,  69,51,  53,108,  116,107,  163,216,  364,343

  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# coco (small cancelled, large, medium, small, yolov3)
n=129 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov4.yaml --bucket ult/coco --name $n && sudo shutdown
n=130 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 32 --weights '' --cfg models/yolov4.yaml --bucket ult/coco --name $n && sudo shutdown
n=133 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 16 --weights '' --cfg models/yolov4.yaml --bucket ult/coco --name $n && sudo shutdown
n=134 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov4s.yaml --bucket ult/coco --name $n && sudo shutdown
n=135 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 32 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n && sudo shutdown
n=136 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 16 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n && sudo shutdown


n=138 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n && sudo shutdown
n=139 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n && sudo shutdown
n=140 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n && sudo shutdown

n=141 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=142 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=143 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=144 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=145 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 12 --weights '' --cfg models/yolov5l.yaml --bucket ult/coco --name $n && sudo shutdown
n=146 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown

n=147 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=148 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=149 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 832 --batch 32 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown

n=150 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 32 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=151 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 320 640 --batch 32 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown

n=152 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n && sudo shutdown
n=154 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 32 --weights '' --cfg models/yolov5s_exp.yaml --bucket ult/coco --name $n && sudo shutdown

n=153 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_focus1.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=155 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_focus2.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=156 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_exp_focus0.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=157 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_focus3.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=158 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_focus4.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=159 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_focus5.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=160 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5_final.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=161 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5_final.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=162 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5_final.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=163 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5_final.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=164 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5_final.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=165 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_origami.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=166 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_k3.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=167 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_plus.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=168 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=169 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=170 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=171 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_k3_spp5913.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=172 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=173 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=174 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown


n=178 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 96 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=179 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 48 --weights '' --cfg models/yolov5m.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=180 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 28 --weights '' --cfg models/yolov5l.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=181 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 16 --weights '' --cfg models/yolov5x.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=182 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=183 && t=ultralytics/coco:v178 && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=184 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=185 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown


n=186 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=187 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=188 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml

n=189 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_focus.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=190 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=191 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml

n=192 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=193 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s.yaml --bucket ult/coco --name $n --data data/coco.yaml

n=194 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=195 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml

n=196 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=197 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp1.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=198 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp2.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=199 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --epochs 50 --batch 64 --weights '' --cfg models/yolov5s_csp2.yaml --bucket ult/coco --name $n --data data/coco.yaml



n=201 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 48 --weights '' --cfg models/yolov5m_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml
n=204 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown

n=206 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown
n=207 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t python3 train.py --img 640 640 --batch 24 --weights '' --cfg models/yolov3-spp_csp.yaml --bucket ult/coco --name $n --data data/coco.yaml && sudo shutdown


n=205 && t=ultralytics/coco:v$n && sudo docker pull $t && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco $t bash
