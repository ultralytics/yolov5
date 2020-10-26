<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/82944393-f7644d80-9f4f-11ea-8b87-1a5b04f555f1.jpg" width="1000"></a>
&nbsp

![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="1000">** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.

- **August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.
- **July 23, 2020**: [v2.0 release](https://github.com/ultralytics/yolov5/releases/tag/v2.0): improved model definition, training and mAP.
- **June 22, 2020**: [PANet](https://arxiv.org/abs/1803.01534) updates: new heads, reduced parameters, improved speed and mAP [364fcfd](https://github.com/ultralytics/yolov5/commit/364fcfd7dba53f46edd4f04c037a039c0a287972).
- **June 19, 2020**: [FP16](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.half) as new default for smaller checkpoints and faster inference [d4c6674](https://github.com/ultralytics/yolov5/commit/d4c6674c98e19df4c40e33a777610a18d1961145).
- **June 9, 2020**: [CSP](https://github.com/WongKinYiu/CrossStagePartialNetworks) updates: improved speed, size, and accuracy (credit to @WongKinYiu for CSP).
- **May 27, 2020**: Public release. YOLOv5 models are SOTA among all known YOLO implementations.
- **April 1, 2020**: Start development of future compound-scaled [YOLOv3](https://github.com/ultralytics/yolov3)/[YOLOv4](https://github.com/AlexeyAB/darknet)-based PyTorch models.


## Pretrained Checkpoints

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** || 7.5M   | 13.2B
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     || 21.8M  | 39.4B
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     || 47.8M  | 88.1B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0)    | **49.2** | **49.2** | **67.7** | 6.9ms     | 145     || 89.0M  | 166.4B
| | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/tag/v3.0) + TTA|**50.8**| **50.8** | **68.9** | 25.5ms    | 39      || 89.0M  | 354.3B
| | | | | | || |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov5/releases/tag/v3.0) | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results in the table denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or test-time augmentation. **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.001`  
** Speed<sub>GPU</sub> measures end-to-end time per image averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) instance with one V100 GPU, and includes image preprocessing, PyTorch FP16 image inference at --batch-size 32 --img-size 640, postprocessing and NMS. Average NMS time included in this chart is 1-2ms/img.  **Reproduce** by `python test.py --data coco.yaml --img 640 --conf 0.1`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce** by `python test.py --data coco.yaml --img 832 --augment` 

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)

## Nanovare inference

If you pass at least one nanovare arguments, detect_nanovare is in nanovare mode (run localization, tracking on the all MAST capture dataset) but is ultralytics-friendly (accept ultralytics arguments).
The nanovare mode only loops on the Mast capture dataset by date and patient and overwrites the ultralytics --source arguments and call the original ultralytics detect.py on this patient dataset.
The nanovare mode performs tasks for each patient under the supervision of luigi so you can quit a detect_nanovare process with no damage. It will start back where it stopped. You can invalidate a task by passing the --invalidate arguments if you wish to rerun a task for any reasons (data changed, task not performed as expected, test, ...)

If using the nanovare mode, you need to set the $PATH_DATA env variable in .env or in the shell to indicate where the capture dataset is.  $PATH_DATA / "capture_MAST_data" is the capture dataset.
AND you need to set the $MAST_ANALYSIS_IDENTIFIER at YOLO to output proper nanovare-friendly localization output (tracking input) inside this capture MAST dataset to avoid conflict with localization output from another MAST method.


If you pass at least one nanovare argument, detect_nanovare is in nanovare mode:

```bash
python detect_nanovare.py --run-tracking    # Run localization for each patient then tracking for each patient (luigi runs the localization task because tracking depends on localization)  
```
```bash
python detect_nanovare.py --run-localization
                          --run-tracking    # Run localization for all patient then tracking for all patient
                    
```
```bash
python detect_nanovare.py --run-tracking    # Run localization for tp23 then tracking patient for tp23
                          --patient-id tp23 # Filter by patient_id
                          --date 2020_05_12 # Filter by date
                          --invalidate      # Invalidate the task TaskRunTracking(patient_id=tp23, date=2020_05_12) before running it again
                                            # Invalidates also automatically all upstream tasks; here the only upstream task 
                                            # to be invalidated is TaskRunLocalization(patient_id=tp23, date=2020_05_12) before running it again
```
```bash
python detect_nanovare.py --run-tracking
                          --patient-id tp23                                                                        # Nanovare arg
                          --date 2020_05_12                                                                        # Nanovare arg
                          --invalidate                                                                             # Nanovare arg
                          --iou-thres 0.8                                                                          # Ultralytics arg
                          --weights ..\..\data\analysis\yolo\minimal_deformation\runs\exp0\weights\weights\best.pt # Ultralytics arg
```
Else if you pass only ultralytics arguments, detect_nanovare is in ultralytics mode

```bash
 python detect_nanovare.py --source  ..\..\data\Karolinska\capture_MAST_data\2020_05_12\test-patient-03 # Ultralytics arg
                           --iou-thres 0.8                                                              # Ultralytics arg
```

All nanovare options:
```bash
(.windows_venv38) Q:\dev\yolov5>python detect_nanovare.py -h
usage: detect_nanovare.py [-h] [--patient-id PATIENT_ID] [--date DATE] [--run-localization] [--run-tracking]
                          [--run-viz] [--invalidate]

optional arguments:
  -h, --help            show this help message and exit
  --patient-id PATIENT_ID
                        Filter a patient ID
  --date DATE           Filter a date
  --run-localization    Run localization
  --run-tracking        Run tracking
  --run-viz             Run vizualization after tracking
  --invalidate          Run vizualization after tracking

```

## Inference

Inference can be run on most common media formats. Model [checkpoints](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J) are downloaded automatically if available. Results are saved to `./inference/output`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
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


## Nanovare Training

If you pass at least one nanovare arguments, detect_nanovare is in nanovare mode (creates, download and check a supervisely dataset, convert it to yolo and train) but is ultralytics-friendly (accept ultralytics arguments).
The nanovare mode creates a new dataset, overwrites the ultralytics --data arguments and call the original ultralytics train.py. This pipeline is tracked thanks to a identifier called 'pipeline_name' that you call as an argument --pipelinename pipeline_name. All datasets and datas related will fall into the 'pipeline_name' folder.
To the contrary of detect_nanovare, the nanovare mode of training does not performs task under the supervision of luigi yet.

If using the nanovare mode, you need to set the $ANALYSIS_PATH_DATA (ex: ../../data/analysis) env variable in .env or in the shell to indicate where the yolo data folder is. $ANALYSIS_PATH_DATA / "yolo" is the yolo data folder.
AND you need to set the supervisely variables SUPERVISELY_PATH_DATA (ex: ../../data/supervisely) for the download folder and SUPERVISELY_API_KEY to set your API token.


If you pass at least one nanovare argument, train is in nanovare mode:

```bash
python train.py strong_mosaic_aug               # specify a pipeline name
                --init-supervisely zoe+vincent  # Download the supervisely dataset if not already
                --init-yolo                     # Convert it to a yolov5-friendly dataset
                --run-train                     # Launch the training on this dataset 
```
```bash
python train.py strong_mosaic_aug                                                             # Nanovare arg
                --run-train                                                                   # Nanovare arg
                --resume ..\..\data\analysis\yolo\strong_mosaic_aug\runs\exp6\weights\last.pt # Ultralytics arg
                    
```

Else if you pass only ultralytics arguments, train is in ultralytics mode

```bash
 python train.py --data  ..\..\data\analysis\yolo\strong_mosaic_aug\data.yaml       # Ultralytics arg
                 --hyp  ..\..\data\analysis\yolo\strong_mosaic_aug\hyp.scratch.yaml # Ultralytics arg
                 --nosave                                                           # Ultralytics arg
                 --notest                                                           # Ultralytics arg
                 --epochs 150                                                       # Ultralytics arg
```

All nanovare options:
```bash
(.windows_venv38) Q:\dev\yolov5>python train.py -h
usage: train.py [-h] [--pipeline-name PIPELINE_NAME] [--init-supervisely {zoe,vincent,zoe+vincent}] [--init-yolo]
                [--run-train] [--gray]

optional arguments:
  -h, --help            show this help message and exit
  --pipeline-name PIPELINE_NAME
                        Name of the pipeline
  --init-supervisely {zoe,vincent,zoe+vincent}
                        Download, check integrity and merge a filtered supervisely dataset
  --init-yolo           Convert a supervisely dataset to a grey|rgb yolo dataset
  --run-train           Train
  --gray                Gray mode

```

### Training

Download [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) and run command below. Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 
