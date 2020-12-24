## Info
This branch provides detection and Android code complement to branch `tf-only-export`.
`models/tf.py` uses TF2 API to construct a tf.Keras model according to `*.yaml` config files and reads weights from `*.pt`, without using ONNX. 

**Because this branch persistently rebases to master branch of ultralytics/yolov5, use `git pull --rebase` instead of `git pull`.**


## Usage
### 1. Git clone `yolov5` and checkout `tf-android`

```
git clone https://github.com/zldrobit/yolov5.git
cd yolov5
git checkout tf-android
```

and download pretrained weights from 
```
https://github.com/ultralytics/yolov5.git

```

### 2. Install requirements
```
pip install -r requirements.txt
pip install tensorflow==2.3.1
```

### 3. Convert and verify
- Convert weights to TensorFlow SavedModel and GraphDef, and verify them with
```
PYTHONPATH=. python models/tf.py --weights weights/yolov5s.pt --cfg models/yolov5s.yaml --img 640
python3 detect.py --weight weights/yolov5s.pb --img 640
python3 detect.py --weight weights/yolov5s_saved_model/ --img 640
```
- Convert weights to TensorFlow SavedModel and GraphDef **integrated with NMS**, and verify them with
```
PYTHONPATH=. python3  models/tf.py --img 640 --weight weights/yolov5s.pt --cfg models/yolov5s.yaml --tf-nms
python3 detect.py --img 640 --weight weights/yolov5s.pb --no-tf-nms
python3 detect.py --img 640 --weight weights/yolov5s_saved_model --no-tf-nms
```
- Convert weights to fp16 TFLite model, and verify it with
```
PYTHONPATH=. python3  models/tf.py --weight weights/yolov5s.pt --cfg models/yolov5s.yaml --img 640 --no-tfl-detect
python3 detect.py --weight weights/yolov5s-fp16.tflite --img 640 --tfl-detect
```
- Convert weights to int8 TFLite model, and verify it with (Post-Training Quantization needs train or val images from [COCO 2017 dataset](https://cocodataset.org/#download))
```
PYTHONPATH=. python3  models/tf.py --weight weights/yolov5s.pt --cfg models/yolov5s.yaml --img 640 --no-tfl-detect --tfl-int8 --source /data/dataset/coco/coco2017/train2017 --ncalib 100
python3 detect.py --weight weights/yolov5s-int8.tflite --img 640 --tfl-int8 --tfl-detect
```

### 4. Put TFLite models in `assets` folder of Android project, and change 
- `inputSize` to `--img`
- `output_width` according to new/old `inputSize` ratio
- `anchors` to `m.anchor_grid` as https://github.com/ultralytics/yolov5/pull/1127#issuecomment-714651073
in android/app/src/main/java/org/tensorflow/lite/examples/detection/tflite/DetectorFactory.java

Then run the program in Android Studio.

If you have further question, plz ask in https://github.com/ultralytics/yolov5/pull/1127
