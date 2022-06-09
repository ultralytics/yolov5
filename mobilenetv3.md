# Yolo v5 + mobilenet v3

## Train

Original code repo [jylink/yolov5-mobilenetv3](https://github.com/jylink/yolov5-mobilenetv3)

run train (for 7 gpus):
```sh
$ python -m torch.distributed.launch --nproc_per_node 7 train.py --batch-size 56 --data coco.yaml --weights  --cfg ./models/yolov5s-mobilenetv3.yaml
```

Issues:

> libgl.so.1: cannot open shared object file:

```sh
$ apt-get install ffmpeg libsm6 libxext6  -y
```

Trained model on COCO dataset https://vlad-n.s3.us-west-1.amazonaws.com/uaav/models/yolov5s-mobilenet.pt

## Eval

Export to OpenVino
```sh
$ cd yolov5
$ python export.py --weights yolov5n.pt --include openvino
$ python export.py --weights yolov5s.pt --include openvino
$ python export.py --weights ./yolov5s-mobilenet.pt --include openvino
```


Validation
```sh
$ cd yolov5
$ python val.py --weights yolov5n_openvino_model --data coco.yaml
$ python val.py --weights yolov5s_openvino_model --data coco.yaml
$ python val.py --weights yolov5s-mobilenet_openvino_model --data coco.yaml
```

