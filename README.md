# Adlik YOLOv5

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.object_detection?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/latest?definitionId=3&branchName=main)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/33433)

Adlik YOLOv5 focuses on the knowledge distillation of the YOLOV5 model and its deployment on on CPU and GPU inference engines, such as OpenVINO.


## Benchmark

- Benchmark in COCO

|  Model  | Compression<br>strategy | Inputs <br>[b, c, h, w] | mAP<sup>val<br>0.5:0.95 | CPU Latency<br>(sync) | CPU Latency<br>(async) | Pretrain weight |
| ------- | ----------------------- | ----------------------- | ----------------------- | --------------------- | ---------------------- | --------------- |
| yolov5s | baseline                | [1, 3, 640, 640]        | 37.2                    | 15.72 ms              | 58.25        ms        | Download        |
| yolov5s | distill                 | [1, 3, 640, 640]        | 39.3                    | 12.24  ms             | 55.53     ms           | Download        |
| yolov5s | INT8                    | [1, 3, 640, 640]        | 36.7                    | 6.56   ms             | 24.07    ms            | Download        |
| yolov5s | distill + INT8          | [1, 3, 640, 640]        | 38.7                    | 6.61     ms           | 24.16    ms            | Download        |

<details>
  <summary>Table Notes (click to expand)</summary>

* all AP results denote COCO val2017 accuracy.
* Intel CPU Performance is measured Intel(R) Xeon(R) Platinum 8260 CPU based on OpenVINO.
* Latency is measured by Openvino's Benchmark tool and use 4 threads for asynchronous inference.
* INT8 quantization uses OpenVINO's POT tool.

</details>

## Install

We recommend using Docker and have provided dockerfile files.

If you need to install it yourself, you need pytorch>=1.7 , python >=3.7 and OpenVINO 2021.4.1.

- Clone the Adlik yolov5 repository and install requirements.txt

```shell
git clone https://github.com/Adlik/yolov5.git
cd yolov5
pip install -r requirements.txt
```

- install OpenVINO (OpenVINO >=2021.4.1)

[Install the Intel Distribution of OpenVIN Toolkit](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino)

## Training

Training and validation reference [yolov5 repo](https://github.com/ultralytics/yolov5/blob/master/README.md)


#### Distillation

```shell
python -m torch.distributed.launch --nproc_per_node 8 train_distillation.py --batch 256 --data coco.yaml --cfg yolov5s.yaml --weights '' --t_weights ./weights/yolov5m.pt --epochs 1000 --device 0,1,2,3,4,5,6,7
```

<details>
  <summary>Notes (click to expand)</summary>

* nproc_per_node specifies how many GPUs you would like to use. We use 8 V100 GPUs.
* batch is the total batch-size. It will be divided evenly to each GPU. After many experiments, we set 256/8=32 per GPU for best performance.

</details>

## Inference in OpenVINO


#### Convert model

- Convert pt to ONNX

```shell
python export.py --weights ./weights/yolov5s.pt --include onnx --opset 13 --simplify
```

- Convert ONNX to IR

```shell

python <INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model weights/yolov5s.onnx --model_name weights/yolov5s-output -s 255 --output Concat_358
```

<details>
  <summary>Notes (click to expand)</summary>

* <INSTALL_DIR> is the path to OpenVINO installation. My default path is /opt/intel/openvino_2021.4.689.
* Concat_358 is the name of the last output node of the ONNX model.

</details>

#### Evaluation

- Evaluate IR model

```shell
python ./deploy/openvino/eval_openvino_yolov5.py  -m ./weights/yolov5s-output.xml
```

- Benchmark

```shell
# Sync mode
python <INSTALL_DIR>/deployment_tools/tools/benchmark_tool/benchmark_app.py -i <INSTALL_DIR>/deployment_tools/demo/car.png -m ./weights/yolov5s-output.xml -d CPU -api sync -progress
```

<details>
  <summary>Notes (click to expand)</summary>

* <INSTALL_DIR> is the path to OpenVINO installation. My default path is /opt/intel/openvino_2021.4.689.
* Test asynchronous latency, set -api = async.

</details>

#### POT

```shell
pot -c ./deploy/openvino/yolov5s_output_pytorch_int8_simple_model.json --output-dir ./weights --progress-bar
```

<details>
  <summary>Notes (click to expand)</summary>

* json file in the deploy/openvino directory.
* For coco, you can also download my calibration images coco_calib from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh.

</details>