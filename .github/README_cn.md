<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>

[English](../README.md) | 简体中文
<div>
   <a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>

<br>
<p>
YOLOv5🚀是一个在COCO数据集上预训练的物体检测架构和模型系列，它代表了<a href="https://ultralytics.com">Ultralytics</a>对未来视觉AI方法的公开研究，其中包含了在数千小时的研究和开发中所获得的经验和最佳实践。
</p>

<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-github.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-linkedin.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-twitter.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-producthunt.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-youtube.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-facebook.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-instagram.png" width="2%" alt="" /></a>
</div>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">文件</div>

请参阅[YOLOv5 Docs](https://docs.ultralytics.com)，了解有关训练、测试和部署的完整文件。

## <div align="center">快速开始案例</div>

<details open>
<summary>安装</summary>

在[**Python>=3.7.0**](https://www.python.org/) 的环境中克隆版本仓并安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)。
```bash
git clone https://github.com/ultralytics/yolov5  # 克隆
cd yolov5
pip install -r requirements.txt  # 安装
```

</details>

<details open>
<summary>推理</summary>

YOLOv5 [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) 推理. [模型](https://github.com/ultralytics/yolov5/tree/master/models) 自动从最新YOLOv5 [版本](https://github.com/ultralytics/yolov5/releases)下载。

```python
import torch

# 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# 图像
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# 推理
results = model(img)

# 结果
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>用 detect.py 进行推理</summary>

`detect.py` 在各种数据源上运行推理, 其会从最新的 YOLOv5 [版本](https://github.com/ultralytics/yolov5/releases) 中自动下载 [模型](https://github.com/ultralytics/yolov5/tree/master/models) 并将检测结果保存到 `runs/detect` 目录。

```bash
python detect.py --source 0  # 网络摄像头
                          img.jpg  # 图像
                          vid.mp4  # 视频
                          path/  # 文件夹
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP 流
```

</details>

<details>
<summary>训练</summary>

以下指令再现了 YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
数据集结果. [模型](https://github.com/ultralytics/yolov5/tree/master/models) 和 [数据集](https://github.com/ultralytics/yolov5/tree/master/data) 自动从最新的YOLOv5 [版本](https://github.com/ultralytics/yolov5/releases) 中下载。YOLOv5n/s/m/l/x的训练时间在V100 GPU上是 1/2/4/6/8天（多GPU倍速）. 尽可能使用最大的 `--batch-size`, 或通过 `--batch-size -1` 来实现 YOLOv5 [自动批处理](https://github.com/ultralytics/yolov5/pull/5092). 批量大小显示为 V100-16GB。

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>教程</summary>

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 推荐
- [获得最佳训练效果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️ 推荐
- [使用 Weights & Biases 记录实验](https://github.com/ultralytics/yolov5/issues/1289)  🌟 新
- [Roboflow：数据集、标签和主动学习](https://github.com/ultralytics/yolov5/issues/4975)  🌟 新
- [多GPU训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)  ⭐ 新
- [TFLite, ONNX, CoreML, TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [测试时数据增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型集成](https://github.com/ultralytics/yolov5/issues/318)
- [模型剪枝/稀疏性](https://github.com/ultralytics/yolov5/issues/304)
- [超参数进化](https://github.com/ultralytics/yolov5/issues/607)
- [带有冻结层的迁移学习](https://github.com/ultralytics/yolov5/issues/1314) ⭐ 新
- [架构概要](https://github.com/ultralytics/yolov5/issues/6998) ⭐ 新

</details>

## <div align="center">环境</div>

使用经过我们验证的环境，几秒钟就可以开始。点击下面的每个图标了解详情。

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
</div>

## <div align="center">如何与第三方集成</div>

<div align="center">
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-long.png" width="49%"/>
    </a>
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow-long.png" width="49%"/>
    </a>
</div>

|Weights and Biases|Roboflow ⭐ 新|
|:-:|:-:|
|通过 [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme) 自动跟踪和可视化你在云端的所有YOLOv5训练运行状态。|标记并将您的自定义数据集直接导出到YOLOv5，以便用 [Roboflow](https://roboflow.com/?ref=ultralytics) 进行训练。 |

<!-- ## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p> -->

## <div align="center">为什么选择 YOLOv5</div>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 图像 (点击扩展)</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>图片注释 (点击扩展)</summary>

- **COCO AP val** 表示 mAP@0.5:0.95 在5000张图像的[COCO val2017](http://cocodataset.org)数据集上，在256到1536的不同推理大小上测量的指标。
- **GPU Speed** 衡量的是在 [COCO val2017](http://cocodataset.org) 数据集上使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100实例在批量大小为32时每张图像的平均推理时间。
- **EfficientDet** 数据来自 [google/automl](https://github.com/google/automl) ，批量大小设置为 8。
- 复现 mAP 方法: `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### 预训练检查点

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n][assets]      |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s][assets]      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m][assets]      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l][assets]      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x][assets]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6][assets]     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6][assets]     |1280 |44.8   |63.7   |385    |8.2    |3.6    |12.6   |16.8
|[YOLOv5m6][assets]     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6][assets]     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4
|[YOLOv5x6][assets]<br>+ [TTA][TTA]|1280<br>1536 |55.0<br>**55.8** |72.7<br>**72.7** |3136<br>- |26.2<br>- |19.4<br>- |140.7<br>- |209.8<br>-

<details>
  <summary>表格注释 (点击扩展)</summary>

- 所有检查点都以默认设置训练到300个时期. Nano和Small模型用 [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, 其他模型使用 [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** 值是 [COCO val2017](http://cocodataset.org) 数据集上的单模型单尺度的值。
<br>复现方法: `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- 使用 [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) 实例对COCO val图像的平均速度。不包括NMS时间（~1 ms/img)
<br>复现方法: `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [测试时数据增强](https://github.com/ultralytics/yolov5/issues/303) 包括反射和比例增强.
<br>复现方法: `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">贡献</div>

我们重视您的意见! 我们希望给大家提供尽可能的简单和透明的方式对 YOLOv5 做出贡献。开始之前请先点击并查看我们的 [贡献指南](CONTRIBUTING.md)，填写[YOLOv5调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) 来向我们发送您的经验反馈。真诚感谢我们所有的贡献者!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->
<a href="https://github.com/ultralytics/yolov5/graphs/contributors"><img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/image-contributors-1280.png" /></a>

## <div align="center">联系</div>

关于YOLOv5的漏洞和功能问题，请访问 [GitHub Issues](https://github.com/ultralytics/yolov5/issues)。商业咨询或技术支持服务请访问[https://ultralytics.com/contact](https://ultralytics.com/contact)。

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-producthunt.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-facebook.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-instagram.png" width="3%" alt="" /></a>
</div>

[assets]: https://github.com/ultralytics/yolov5/releases
[tta]: https://github.com/ultralytics/yolov5/issues/303
