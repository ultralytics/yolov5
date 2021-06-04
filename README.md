# [YOLOv5](https://ultralytics.com/yolov5) 🚀 by [Ultralytics](https://ultralytics.com)

<div align="center">
<p>
<img src="assets/img/yolov5-logo-s.png" />
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
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset.
</p>
</div>  

*Note : YOLOv5 is current **under active development**, all code, models, and documentation are subject to modification or deletion without notice. __Use at your own risk.__*

## <div align="center">Documentation</div>
* [Full Documentation](https://docs.ultralytics.com)

## <div align="center">Quick Start Tutorials</div>
These tutorials are intended to get you started using YOLOv5 quickly for demonstration purposes. Head to the [Full Documentation](https://docs.ultralytics.com) for more indept tutorials. You can test YOLOv5 with [Ultralytics API](https://ultralytics.com/yolov5), no coding required.
<details>
<summary>
Install with Docker
</summary>  

```bash
sudo docker pull ultralytics/yolov5:latest
```
find more details in our [docker tutorial](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart)
</details>
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

*NOTE : In order to follow this tutorial please ensure you have installed YOLOv5 locally.*  

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

*NOTE : In order to follow this tutorial please ensure you have installed YOLOv5 locally.*  

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16

```

</details>

  
## <div align="center">Easy integrations</div>
YOLOv5 🚀 integrates easily with <b>[Wand & Biases](https://github.com/ultralytics/yolov5/issues/1289)</b> and [Supervise.ly](https://github.com/ultralytics/yolov5/issues/2518). Integration with [AWS](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart) and [Google cloud](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) takes only few minutes.
  
  <br/>

  ## <div align="center">Contribute and win</div>
  <p>
  We are super excited to announce the first-ever Ultralytics YOLOv5 rocket EXPORT Competition with $10,000.00 in cash prizes!!!
</p> 
<p> 
  <a src="https://github.com/ultralytics/yolov5/discussions/3213"><img src="assets/img/export_competition_banner.png" /></a>
</p> 
<br/>


## <div align="center">Why YOLOv5</div>

Because we improve every day, we constantly listen to you and bring you a solution that makes your work easier

<div align="center">
<img src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png">
</div>


<br/>


### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPS<br><sup>640 (B)
---   |---  |---        |---         |---             |---                |---|---              |---
[YOLOv5s][assets]    |640  |36.7     |36.7     |55.4     |**2.0** | |7.3   |17.0
[YOLOv5m][assets]    |640  |44.5     |44.5     |63.1     |2.7     | |21.4  |51.3
[YOLOv5l][assets]    |640  |48.2     |48.2     |66.9     |3.8     | |47.0  |115.4
[YOLOv5x][assets]    |640  |**50.4** |**50.4** |**68.8** |6.1     | |87.7  |218.8
| | | | | | || |
[YOLOv5s6][assets]   |1280 |43.3     |43.3     |61.9     |**4.3** | |12.7  |17.4
[YOLOv5m6][assets]   |1280 |50.5     |50.5     |68.7     |8.4     | |35.9  |52.4
[YOLOv5l6][assets]   |1280 |53.4     |53.4     |71.1     |12.3    | |77.2  |117.7
[YOLOv5x6][assets]   |1280 |**54.4** |**54.4** |**72.0** |22.4    | |141.8 |222.9
| | | | | | || |
[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8 | |-  |-



<br/>

## <div align="center">Getting Involved</div>

We are growing quickly and have new [positions](https://startupmatcher.com/s/ultralytics) opened. Be a part of our team!
<br/>  
Contributors should raise issues directly in the repository. 

## <div align="center">Get in Touch</div>

For business or professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact)
