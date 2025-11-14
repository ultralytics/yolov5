# ConvNeXt Backbones in YOLO for Object Detection

This project integrates modern ConvNeXt backbones into the YOLOv5 framework to improve object detection performance, particularly for small or densely packed objects. We replace YOLOv5's default CSPDarknet backbone with ConvNeXt-Tiny while maintaining YOLO's detection head and neck architecture.

## Project Overview

**Goal:** Analyze whether architectural advances in ConvNeXt can improve YOLO's detection capability by systematically quantifying how a ConvNeXt backbone changes detection performance on standardized benchmarks.

**Approach:**

- Use YOLOv5s as baseline (easier to modify than YOLOv8)
- Replace CSPDarknet backbone with ConvNeXt-Tiny from [timm library](https://github.com/huggingface/pytorch-image-models)
- Load pretrained ConvNeXt weights from [HuggingFace](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k)
- Initialize YOLO neck with random weights for fine-tuning
- Ensure architectural compatibility by aligning feature map dimensions

## Dataset

We use the MS-COCO dataset containing:

- ~118,000 training images
- ~5,000 validation images
- 80 object categories with bounding box annotations
- Images resized to 640×640 pixels with standard YOLO augmentations

**Dataset Location (SCC):** `/projectnb/ec523bn/projects/ConvNeXt-YOLO/`

## Evaluation Metrics

Performance measured using:

- **mAP@[.5:.95]:** Mean Average Precision over IoU thresholds from 0.5 to 0.95
- **mAP@0.5:** Mean Average Precision at IoU threshold of 0.5
- **Precision & Recall:** Standard detection metrics

## Baseline Results

YOLOv5s fine-tuned for 50 epochs on MS-COCO:

| Metric       | Official YOLOv5s | Our Baseline |
| ------------ | ---------------- | ------------ |
| mAP@0.5:0.95 | 0.374            | **0.370**    |
| mAP@0.5      | 0.572            | **0.568**    |
| Precision    | 0.672            | **0.651**    |
| Recall       | 0.519            | **0.521**    |

## Setup & Training

### Prerequisites

- PyTorch
- YOLOv5 dependencies
- Access to COCO dataset

### Training

1. In `coco.yaml`, change the "path" variable to the path of the dataset. The COCO2017 Dataset is stored on SCC at `/projectnb/ec523bn/projects/ConvNeXt-YOLO/datasets`. In this case, change the path to `/projectnb/ec523bn/projects/ConvNeXt-YOLO/datasets/coco`.

2. Run the training script with the following arguments:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data coco.yaml --weights yolov5s.pt --cache disk
```

Notable changes from defaults:

- Using `coco.yaml` instead of `coco128.yaml`
- Using `--cache disk` instead of the RAM default because of the large dataset
- Training for 50 epochs to establish baseline

### ConvNeXt Integration (In Progress)

Key code changes required:

- Custom backbone class for ConvNeXt-Tiny
- Modified model YAML configuration
- Feature map alignment layers for channel compatibility
- Selective weight loading logic (ConvNeXt pretrained + random neck weights)
- Custom training flags for backbone selection

## Project Structure

This is a fork of the official [YOLOv5 repository](https://github.com/ultralytics/yolov5) with modifications to support ConvNeXt backbones.

## References

1. Liu et al. "A ConvNet for the 2020s" (ConvNeXt), arXiv 2022
2. Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection", arXiv 2016
3. Terven et al. "A Comprehensive Review of YOLO Architectures", Machine Learning and Knowledge Extraction, 2023
4. Topuz et al. "ConvNeXt Mitosis Identification—YOLO", Laboratory Investigation, 2024
5. Jiang et al. "ConvNeXt-YOLO Architecture with Multi-Scale Signal Processing", IEEE AUTEEE, 2024
6. Lin et al. "Microsoft COCO: Common Objects in Context", ECCV 2014

## Logging

All training experiments are logged via [Comet](https://www.comet.com/site/) for comprehensive model evaluation and tracking.

## Success Criteria

The project will be considered successful if the ConvNeXt-backed YOLO model achieves equal or higher mAP than the baseline YOLOv5s model under identical training conditions.
