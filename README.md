# HIC-YOLOv5: Improved YOLOv5 for Small Object Detection

## Overview

This repository contains the code for HIC-YOLOv5, an improved version of YOLOv5 tailored for small object detection. The improvements are based on the paper [HIC-YOLOv5: Improved YOLOv5 For Small Object Detection](https://arxiv.org/pdf/2309.16393v1.pdf).

HIC-YOLOv5 incorporates Channel Attention Block (CBAM) and Involution modules for enhanced object detection, making it suitable for both CPU and GPU training.

## Installation

The installation process for HIC-YOLOv5 is identical to the YOLOv5 repository. You can follow the installation instructions provided in the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

## Usage

To use HIC-YOLOv5, you can specify the configuration file with the `--cfg` argument. An example command for training might look like this:

```bash
python train.py --img-size 640 --batch 16 --epochs 100 --data data/coco.yaml --cfg models/yolo5m-cbam-involution.yaml
```

- `--img-size`: Specifies the input image size.
- `--batch`: Sets the batch size for training.
- `--epochs`: Defines the number of training epochs.
- `--data`: Specifies the data configuration file.
- `--cfg`: Points to the configuration file for HIC-YOLOv5. In this case, it's the `models/yolo5m-cbam-involution.yaml`.

## Testing for Multi-GPU Training (TODO)

I am actively working on adding support for multi-GPU training. Please stay tuned for updates on testing and training with multiple GPUs.

## Acknowledgments

I want to express our gratitude to the authors of the paper "HIC-YOLOv5: Improved YOLOv5 For Small Object Detection" for their contributions, which inspired the development of HIC-YOLOv5.

## License

HIC-YOLOv5 is released under the MIT License. Please refer to the LICENSE file for more details.

For additional information and updates, please refer to the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

**Note:** Be sure to refer to the official [YOLOv5 repository](https://github.com/ultralytics/yolov5) for the latest updates and documentation.
