## Info
This branch provides EdgeTPU support complement to branch `tf-only-export`.
`models/tf.py` uses TF2 API to construct a tf.Keras model according to `*.yaml` config files and reads weights from `*.pt`, without using ONNX. 

**Because this branch persistently rebases to master branch of ultralytics/yolov5, use `git pull --rebase` instead of `git pull`.**

## Usage
### 1. Git clone `yolov5` and checkout `tf-edgetpu`
```
git clone https://github.com/zldrobit/yolov5.git
cd yolov5
git checkout tf-edgetpu
```
and download pretrained weights from 
```
https://github.com/ultralytics/yolov5.git
```

### 2. Install requirements
```
pip install -r requirements.txt
pip install tensorflow==2.4.1
```

### 3. Convert and verify
- Convert weights to int8 TFLite model, and verify it with (Post-Training Quantization needs train or val images from [COCO 2017 dataset](https://cocodataset.org/#download))
```
python3 models/tf.py --weights weights/yolov5s.pt --cfg models/yolov5s.yaml --img 320 --no-tfl-detect --tfl-int8 --tf-raw-resize --source /data/dataset/coco/coco2017/train2017 --ncalib 100
python3 detect.py --weight weights/yolov5s-int8.tflite --img 320 --tfl-int8 --tfl-detect
```
- Convert full int8 TFLite model to **Edge TPU** and verify it with
```
# need Edge TPU runtime https://coral.ai/software/#edgetpu-runtime
# and Edge TPU compiler https://coral.ai/software/#debian-packages
edgetpu_compiler -s -a -o weights weights/yolov5s-int8.tflite
python3 detect.py --weights weights/yolov5s-int8_edgetpu.tflite --edgetpu --tfl-int8 --tfl-detect --img 320
```

If you have further question, plz ask in https://github.com/ultralytics/yolov5/pull/3630

