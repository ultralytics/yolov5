# [YOLOv5](https://ultralytics.com/yolov5) by [Ultralytics](https://ultralytics.com)

<div align="center">
<p>
<img src="assets/img/yolo-header.gif" />
</p>
<br>
<div>
<a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Open In Kaggle"></a>
<br>  
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
</div>

<br>
<p>
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset. This repository represents Ultralytics open-source research into future object detection methods, and incorporates lessons learned and best practices evolved over thousands of hours of training and evolution on anonymized client datasets.
</p>
</div>

### <div align="center">[See YOLOv5 in Action with Our Interactive Demo Here](https://ultralytics.com/yolov5)</div>

_Note : YOLOv5 is current **under active development**, all code, models, and documentation are subject to modification or deletion without notice. **Use at your own risk.**_

## <div align="center">Documentation</div>

Check out our [Full Documentation](https://docs.ultralytics.com) or use our Quick Start Tutorials.

## <div align="center">Quick Start Tutorials</div>

These tutorials are intended to get you started using YOLOv5 quickly for demonstration purposes.  
Head to the [Full Documentation](https://docs.ultralytics.com) for more in-depth tutorials.

<details>
<summary>
Install Locally
</summary>

```bash
# Clone into current directory
$ git clone git@github.com:ultralytics/yolov5.git .
# Install requirements
$ pip install -r requirements.txt
```

</details>
<details>
<summary>Inference Using Repository Clone</summary>

_NOTE : In order to follow this tutorial please ensure you have installed YOLOv5 locally._

```bash
# Run inference based on selected input
$ python detect.py --source 0  # webcam
                            file.jpg  # image
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>
<details open>
<summary>Inference Using PyTorch Hub</summary>

This tutorial will automatically download YOLOv5 to your local system before running inference on the supplied image.

```python
import torch

# Define your model, options include yolov5s, yolov5m, yolov5l, yolov5x
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define your image
img = 'https://ultralytics.com/images/zidane.jpg'

# Run inference
results = model(img)

# Handle your results, options include .print(), .show(), .save(), .panadas().xyz()
results.print()
```

</details>

<details>
<summary>Training</summary>

_NOTE : In order to follow this tutorial please ensure you have installed YOLOv5 locally._

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16

```

</details>  

## <div align="center">Environments and Integrations</div>

Get started with YOLOv5 in less than a few minutes using our integrations.  

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="assets/img/agents/colab-small.svg" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="assets/img/agents/kaggle-small.svg" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="assets/img/agents/docker-small.svg" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="assets/img/agents/aws-small.svg" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="assets/img/agents/gcp-small.svg" width="15%"/>
    </a>
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_wandbtutorial">
        <img src="assets/img/agents/wb-small.svg" width="15%"/>
    </a>
</div>  



Add these your toolkit to ensure you get the most out of your training experience:  

* [Weight and Biasis](https://wandb.ai/site?utm_campaign=repo_yolo_wandbtutorial) - Debug, compare and reproduce models. Easily visualize performance with powerful custom charts.
* [Supervisely](https://app.supervise.ly/signup) - Data labeling for images, videos, 3D point cloud, and volumetric medical images  

## <div align="center">Contribue and Win</div>

We are super excited to announce our first-ever Ultralytics YOLOv5 rocket EXPORT Competition with **$10,000** in cash prizes!  

<div align="center">
<a href="https://github.com/ultralytics/yolov5/discussions/3213">
    <img src="assets/img/export_competition_banner.png"/>
</a>
</div>

## <div align="center">Why YOLOv5</div>

<div align="center">

**Its Fast!**  
**Its Accurate!**  
**But above all YOLOv5 is super easy to get up and running due to its PyTorch integration.**

</div>

<br>
<div align="center">
<img src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png">
</div>

### Pretrained Checkpoints

| Model                  | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>test<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>V100 (ms) |     | params<br><sup>(M) | FLOPS<br><sup>640 (B) |
| ---------------------- | --------------------- | ----------------------- | ------------------------ | ------------------ | ----------------------- | --- | ------------------ | --------------------- |
| [YOLOv5s][assets]      | 640                   | 36.7                    | 36.7                     | 55.4               | **2.0**                 |     | 7.3                | 17.0                  |
| [YOLOv5m][assets]      | 640                   | 44.5                    | 44.5                     | 63.1               | 2.7                     |     | 21.4               | 51.3                  |
| [YOLOv5l][assets]      | 640                   | 48.2                    | 48.2                     | 66.9               | 3.8                     |     | 47.0               | 115.4                 |
| [YOLOv5x][assets]      | 640                   | **50.4**                | **50.4**                 | **68.8**           | 6.1                     |     | 87.7               | 218.8                 |
|                        |                       |                         |                          |                    |                         |     |                    |
| [YOLOv5s6][assets]     | 1280                  | 43.3                    | 43.3                     | 61.9               | **4.3**                 |     | 12.7               | 17.4                  |
| [YOLOv5m6][assets]     | 1280                  | 50.5                    | 50.5                     | 68.7               | 8.4                     |     | 35.9               | 52.4                  |
| [YOLOv5l6][assets]     | 1280                  | 53.4                    | 53.4                     | 71.1               | 12.3                    |     | 77.2               | 117.7                 |
| [YOLOv5x6][assets]     | 1280                  | **54.4**                | **54.4**                 | **72.0**           | 22.4                    |     | 141.8              | 222.9                 |
|                        |                       |                         |                          |                    |                         |     |                    |
| [YOLOv5x6][assets] TTA | 1280                  | **55.0**                | **55.0**                 | **72.0**           | 70.8                    |     | -                  | -                     |

<br>

## <div align="center">Getting Involved and Contributing</div>

Please make sure to read the [Contributing Guide](CONTRIBUTING.md) before making a pull request.

**Thank you to all the people who already contributed to YOLOv5!**

Issues should be raised in [GitHub Issues](https://github.com/ultralytics/yolov5/issues) provided yours does not already exist.

## <div align="center">Get in Touch</div>

**For issues or trouble running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues) and create a new issue provided yours does not already exist.**
<br>  
For business or professional support requests please visit:  
[https://ultralytics.com/contact](https://ultralytics.com/contact)

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="assets/img/social-media/github.svg" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="assets/img/social-media/linkedin.svg" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="assets/img/social-media/twitter.svg" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="assets/img/social-media/youtube.svg" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="assets/img/social-media/facebook.svg" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="assets/img/social-media/instagram.svg" width="3%"/>
    </a>
</div>
