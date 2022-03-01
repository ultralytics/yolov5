# Adlik YOLOv5

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.object_detection?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/latest?definitionId=3&branchName=main)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/33433)

Adlik YOLOv5 focuses on the knowledge distillation of the YOLOV5 model and deployment on Intel CPU with OpenVINO inference engine.

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch and more
- Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud


## Benchmark

**Benchmark in COCO**

|  Model  |   Compression<br>strategy   | Input size <br>[h, w] | mAP<sup>val<br>0.5:0.95 |                                                                                   Pretrain weight                                                                                    |
| ------- | --------------------------- | --------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| yolov5s | baseline                    | [640, 640]            | 37.2                    | [pth](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt) &#124; [onnx](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s.onnx)                          |
| yolov5s | distillation                | [640, 640]            | 39.3                    | [pth](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.pt) &#124; [onnx](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.onnx)            |
| yolov5s | quantization                | [640, 640]            | 36.5                    | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.bin)                 |
| yolov5s | distillation + quantization | [640, 640]            | 38.6                    | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.bin) |


**CPU Performance is measured Intel(R) Xeon(R) Platinum 8260 CPU based on OpenVINO**

|  Model  |    Type    | Inputs <br>[b, c, h, w] | CPU Latency<br>(sync) | CPU Latency<br>(async) | CPU FPS <br>(sync) | CPU FPS <br>(async) |
| ------- | ---------- | ----------------------- | --------------------- | ---------------------- | ------------------ | ------------------- |
| yolov5s | FP32, onnx | [1, 3, 640, 640]        | 15.72   ms            | 58.25      ms          | 63.61              | 204.19              |
| yolov5s | FP32, IR   | [1, 3, 640, 640]        | 12.95   ms            | 55.53      ms          | 77.22              | 215.39              |
| yolov5s | INT8, IR   | [1, 3, 640, 640]        | 6.56     ms           | 24.07       ms         | 152.51             | 497.28              |

Table Notes:

* All mAP results denote COCO val2017 accuracy.
* Latency is measured by Openvino's Benchmark tool and uses 12 threads for asynchronous inference.
* INT8 quantization uses OpenVINO's POT tool.
* Download pre-trained models and find more models in Model Zoo
* yolov5s.pt is the yolov5 v6.0.

## Update

2022.02.24

- Support knowledge distillation of the YOLOv5 model.
- Add evaluation of OpenVINO IR model.
- Added configuration JSON files using POT quantization.

## Install

We recommend using Docker and have provided dockerfile files.

If you need to install it yourself, you need pytorch>=1.7, python >=3.7, and OpenVINO 2021.4.1.

1. Clone the Adlik yolov5 repository and install requirements.txt

```shell
git clone https://github.com/Adlik/yolov5.git
cd yolov5
pip install -r requirements.txt
```

2. Install OpenVINO (OpenVINO >=2021.4.1)

[Install the Intel Distribution of OpenVINO Toolkit](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino)

## Training

YOLOv5 model training and validation reference [yolov5 repo](https://github.com/ultralytics/yolov5/blob/master/README.md)

#### Knowledge distillation

COCO dataset is more difficult to be used as the training target of object detection task, which means that the teacher network will predict more background bbox. If the prediction output of teacher network is directly used as the soft label of student network learning, there will be a serious problem of class imbalance. In order to solve these problems, we use the distillation method in the paper "Object Detection at 200 Frames Per Second".

```shell
python -m torch.distributed.launch --nproc_per_node 8 train_distillation.py --batch 256 --data coco.yaml --cfg yolov5s.yaml --weights '' --t_weights ./weights/yolov5m.pt --epochs 1000 --device 0,1,2,3,4,5,6,7
```

Notes :

* nproc_per_node specifies how many GPUs you would like to use. We use 8 V100 GPUs.
* batch is the total batch-size. It will be divided evenly for each GPU. After many experiments, we set 256/8=32 per GPU for best performance.


## Inference in OpenVINO

#### Convert model

Convert pt to ONNX:

```shell
python export.py --weights ./weights/yolov5s.pt --include onnx --opset 13 --simplify
```

Convert ONNX to IR:

```shell
python <INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model weights/yolov5s.onnx --model_name weights/yolov5s-output -s 255 --output Concat_358
```

Notes:

* <INSTALL_DIR> is the path to OpenVINO installation. My default path is /opt/intel/openvino_2021.4.689.
* Concat_358 is the name of the last output node of the ONNX model.


#### Evaluation

**Evaluate IR model:**

```shell
python ./deploy/openvino/eval_openvino_yolov5.py  -m ./weights/yolov5s.xml
```

**Benchmark tool:**

```shell
# Sync mode
python <INSTALL_DIR>/deployment_tools/tools/benchmark_tool/benchmark_app.py -i <INSTALL_DIR>/deployment_tools/demo/car.png -m ./weights/yolov5s.xml -d CPU -api sync -progress
```

Notes:

* <INSTALL_DIR> is the path to OpenVINO installation. My default path is /opt/intel/openvino_2021.4.689.
* Test asynchronous latency, set -api = async.

#### POT

We use the defaultQuantization algorithm that designed to perform a fast and in many cases accurate 8-bits quantization of NNs.

```shell
pot -c ./deploy/openvino/yolov5s_output_pytorch_int8_simple_model.json --output-dir ./weights --progress-bar
```

Notes:

* JSON file in the deploy/openvino directory.
* For coco, you can also download calibration images coco_calib from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh.

## Acknowledgements

- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)