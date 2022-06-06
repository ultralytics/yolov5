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
