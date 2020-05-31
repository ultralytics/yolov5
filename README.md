<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.
<img src="https://user-images.githubusercontent.com/26833433/83359175-63b6c680-a32d-11ea-970a-9f602e022468.png" width="800">

Updates:
- **May 27, 2020**: Public release of repo. yolov3-spp implementation (this repo) is SOTA at 45.5 mAP among all known yolo implementations, yolov5 family will be undergoing architecture research and development over Q2/Q3 2020 to increase performance. Updates may include [CSP](https://github.com/WongKinYiu/CrossStagePartialNetworks) bottlenecks from [yolov4](https://github.com/AlexeyAB/darknet), as well as PANet or BiFPN head features.
- **May 24, 2020**: Training yolov5s/x and yolov3-spp. yolov5m/l suffered early overfitting and also code 137 early docker terminations, cause unknown. yolov5l underperforms yolov3-spp due to earlier overfitting, cause unknown.
- **April 1, 2020**: Begin development of a 100% pytorch scaleable yolov3/4-based group of future models, in small, medium, large and extra large sizes, collectively known as yolov5. Models will be defined by new user-friendly yaml-based configuration files for ease of construction and modification. Datasets will likewise use yaml configuration files. New training platform will be simpler use, harder to break, and more robust to training a wider variety of custom dataset.


## Ultralytics Professional Support

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** surveillance systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Pretrained Checkpoints

|       Model    |  AP<sup>val</sup> | AP<sup>test</sup>    |  AP<sub>50</sub> | Latency<sub>GPU</sub> | FPS<sub>GPU</sub>  | | params | FLOPs |
|----------     |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
|     YOLOv5-s ([ckpt](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))    | 33.1 | 33.0 | 53.3 | **3.3ms** | **303** | | 7.0M   | 14.0B
|     YOLOv5-m ([ckpt](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))    | 41.5 | 41.5 | 61.5 | 5.5ms | 182 | | 25.2M  | 50.2B
|     YOLOv5-l ([ckpt](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))    | 44.2 | 44.5 | 64.3 | 9.7ms | 103 | | 61.8M  | 123.1B
|     YOLOv5-x ([ckpt](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))    | **47.1** | **47.2** | **66.7** | 15.8ms | 63 | | 123.1M | 245.7B
|     YOLOv3-SPP ([ckpt](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))  | 45.5 | 45.4 | 65.2 | 8.9ms | 112 | | 63.0M  | 118.0B

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results in the table denote val2017 accuracy.  
** All accuracy numbers are for single-model single-scale without ensemble or test-time augmentation. Reproduce by  `python test.py --img-size 736 --conf_thres 0.001`  
** Latency<sub>GPU</sub> measures end-to-end latency per image averaged over 5000 COCO val2017 images using a V100 GPU and includes image preprocessing, inference, postprocessing and NMS. Average NMS time included in this chart is 1.6ms/image.  Reproduce by `python test.py --img-size 640 --conf_thres 0.1 --batch-size 16`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) with training, testing and testing examples
* [GCP Quickstart](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
* [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) 


## Inference

Inference can be run on most common media formats. Model [checkpoints](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J) are downloaded automatically if available. Results are saved to `./inference/output`.
```bash
$ python detect.py --source file.jpg  # image 
                            file.mp4  # video
                            ./dir  # directory
                            0  # webcam
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on examples in the `./inference/images` folder:

```bash
$ python detect.py --source ./inference/images/ --weights yolov5s.pt --conf 0.4

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', fourcc='mp4v', half=False, img_size=640, iou_thres=0.5, output='inference/output', save_txt=False, source='./inference/images/', view_img=False, weights='yolov5s.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', total_memory=16280MB)

Downloading https://drive.google.com/uc?export=download&id=1R5T6rIyy3lLwgFXNms8whc-387H0tMQO as yolov5s.pt... Done (2.6s)

image 1/2 inference/images/bus.jpg: 640x512 3 persons, 1 buss, Done. (0.009s)
image 2/2 inference/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.009s)
Results saved to /content/yolov5/inference/output
```

<img src="https://user-images.githubusercontent.com/26833433/83082816-59e54880-a039-11ea-8abe-ab90cc1ec4b0.jpeg" width="500">  


## Reproduce Our Training

Run commands below. Training takes a few days for yolov5s, to a few weeks for yolov5x on a 2080Ti GPU.
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 
```
<img src="https://user-images.githubusercontent.com/26833433/82960433-5a191180-9f6f-11ea-85cc-c49dbd1555e1.png" width="900">


## Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **GCP** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Google Colab Notebook** with 12 hours of free GPU time: [Google Colab Notebook](https://colab.sandbox.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)
- **Docker Image** from https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) 


## Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit us at https://www.ultralytics.com.
