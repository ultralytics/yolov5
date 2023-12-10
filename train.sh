python3 train.py \
    --cfg models/yolov5s.yaml --weights weights/yolov5s.pt \
    --data resource/shipdata-yolov5/data.yaml \
    --batch-size 16 --epochs 100 \
    --device 0

