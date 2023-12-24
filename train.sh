python3 train.py \
    --cfg models/yolov5s.yaml --weights weights/yolov5s.pt \
    --data resource/ships-yolov5/data.yaml \
    --batch-size 32 --epochs 30 \
    --device 0

