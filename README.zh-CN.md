<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="850" src="https://raw.githubusercontent.com/ultralytics/assets/master/yolov5/v70/splash.png"></a>
  </p>

[英语](README.md)|[简体中文](README.zh-CN.md)<br>

<div>
    <a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

YOLOv5 🚀 是世界上最受欢迎的视觉 AI，代表<a href="https://ultralytics.com">超力</a>对未来视觉 AI 方法的开源研究，结合在数千小时的研究和开发中积累的经验教训和最佳实践。

要申请企业许可证，请填写表格<a href="https://ultralytics.com/license">Ultralytics 许可</a>.

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
</div>

## <div align="center">Ultralytics 现场会议</div>

<div align="center">

[Ultralytics Live Session Ep。 2个](https://youtu.be/LKpuzZllNpA)✨将直播**欧洲中部时间 12 月 13 日星期二 19:00**和[约瑟夫·纳尔逊](https://github.com/josephofiowa)的[机器人流](https://roboflow.com/?ref=ultralytics)谁将与我们一起讨论全新的 Roboflow x Ultralytics HUB 集成。收听 Glenn 和 Joseph 询问如何通过无缝数据集集成来加快工作流程！ 🔥

<a align="center" href="https://youtu.be/LKpuzZllNpA" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/85292283/205996456-bf3efa33-9c46-455e-b322-a64886cc7a0b.png"></a>
</div>

## <div align="center">细分 ⭐ 新</div>

<div align="center">
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/61612323/204180385-84f3aca9-a5e9-43d8-a617-dda7ca12e54a.png"></a>
</div>

我们新的 YOLOv5[发布 v7.0](https://github.com/ultralytics/yolov5/releases/v7.0)实例分割模型是世界上最快和最准确的，击败所有当前[SOTA 基准](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco).我们使它们非常易于训练、验证和部署。查看我们的完整详细信息[发行说明](https://github.com/ultralytics/yolov5/releases/v7.0)并访问我们的[YOLOv5 分割 Colab 笔记本](https://github.com/ultralytics/yolov5/blob/master/segment/tutorial.ipynb)快速入门教程。

<details>
  <summary>Segmentation Checkpoints</summary>

<br>

我们使用 A100 GPU 在 COCO 上以 640 图像大小训练了 300 个时期的 YOLOv5 分割模型。我们将所有模型导出到 ONNX FP32 以进行 CPU 速度测试，并导出到 TensorRT FP16 以进行 GPU 速度测试。我们在 Google 上进行了所有速度测试[协作临](https://colab.research.google.com/signup)便于重现的笔记本。

| 模型                                                                                         | 尺寸<br><sup>（像素） | 地图<sup>盒子<br>50-95 | 地图<sup>面具<br>50-95 | 火车时间<br><sup>300个纪元<br>A100（小时） | 速度<br><sup>ONNX 中央处理器<br>（小姐） | 速度<br><sup>同仁堂A100<br>（小姐） | 参数<br><sup>(男) | 失败者<br><sup>@640（二） |
| ------------------------------------------------------------------------------------------ | --------------- | ------------------ | ------------------ | ------------------------------- | ----------------------------- | -------------------------- | -------------- | ------------------- |
| [YOLOv5n-se](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt)  | 640             | 27.6               | 23.4               | 80:17                           | **62.7**                      | **1.2**                    | **2.0**        | **7.1**             |
| [YOLOv5s-se](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt)  | 640             | 37.6               | 31.7               | 88:16                           | 173.3                         | 1.4                        | 7.6            | 26.4                |
| [YOLOv5m段](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt)    | 640             | 45.0               | 37.1               | 108:36                          | 427.0                         | 2.2                        | 22.0           | 70.8                |
| [YOLOv5l-se](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt)  | 640             | 49.0               | 39.9               | 我：43（X）                         | 857.4                         | 2.9                        | 47.9           | 147.7               |
| [YOLOv5x-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt) | 640             | **50.7**           | **41.4**           | 62:56 (zks)                     | 1579.2                        | 4.5                        | 88.8           | 265.7               |

- 使用 SGD 优化器将所有检查点训练到 300 个时期`lr0=0.01`和`weight_decay=5e-5`在图像大小 640 和所有默认设置。<br>运行记录到[HTTPS://玩豆瓣.爱/Glenn-就ocher/yo lo V5_V70_official](https://wandb.ai/glenn-jocher/YOLOv5_v70_official)
- **准确性**值适用于 COCO 数据集上的单模型单尺度。<br>重现者`python segment/val.py --data coco.yaml --weights yolov5s-seg.pt`
- **速度**使用 a 对超过 100 个推理图像进行平均[协作临](https://colab.research.google.com/signup)A100 高 RAM 实例。值仅表示推理速度（NMS 每张图像增加约 1 毫秒）。<br>重现者`python segment/val.py --data coco.yaml --weights yolov5s-seg.pt --batch 1`
- **出口**到 FP32 的 ONNX 和 FP16 的 TensorRT 完成`export.py`.<br>重现者`python export.py --weights yolov5s-seg.pt --include engine --device 0 --half`

</details>

<details>
  <summary>Segmentation Usage Examples &nbsp;<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/segment/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></summary>

### 火车

YOLOv5分割训练支持自动下载COCO128-seg分割数据集`--data coco128-seg.yaml`COCO-segments 数据集的参数和手动下载`bash data/scripts/get_coco.sh --train --val --segments`接着`python train.py --data coco.yaml`.

```bash
# Single-GPU
python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3
```

### 瓦尔

在 COCO 数据集上验证 YOLOv5s-seg mask mAP：

```bash
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate
```

### 预测

使用预训练的 YOLOv5m-seg.pt 来预测 bus.jpg：

```bash
python segment/predict.py --weights yolov5m-seg.pt --data data/images/bus.jpg
```

```python
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5m-seg.pt"
)  # load from PyTorch Hub (WARNING: inference not yet supported)
```

| ![zidane](https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg) | ![bus](https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg) |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |

### 出口

将 YOLOv5s-seg 模型导出到 ONNX 和 TensorRT：

```bash
python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0
```

</details>

## <div align="center">文档</div>

见[YOLOv5 文档](https://docs.ultralytics.com)有关培训、测试和部署的完整文档。请参阅下面的快速入门示例。

<details open>
<summary>Install</summary>

克隆回购并安装[要求.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)在一个[**Python>=3.7.0**](https://www.python.org/)环境，包括[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>Inference</summary>

YOLOv5[PyTorch 中心](https://github.com/ultralytics/yolov5/issues/36)推理。[楷模](https://github.com/ultralytics/yolov5/tree/master/models)自动从最新下载
YOLOv5[发布](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py`在各种来源上运行推理，下载[楷模](https://github.com/ultralytics/yolov5/tree/master/models)自动从
最新的YOLOv5[发布](https://github.com/ultralytics/yolov5/releases)并将结果保存到`runs/detect`.

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

下面的命令重现 YOLOv5[可可](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)结果。[楷模](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)自动从最新下载
YOLOv5[发布](https://github.com/ultralytics/yolov5/releases). YOLOv5n/s/m/l/x 的训练时间为
V100 GPU 上 1/2/4/6/8 天（[多GPU](https://github.com/ultralytics/yolov5/issues/475)倍快）。使用
最大的`--batch-size`可能，或通过`--batch-size -1`为了
YOLOv5[自动批处理](https://github.com/ultralytics/yolov5/pull/5092).显示的批量大小适用于 V100-16GB。

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>Tutorials</summary>

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)🚀 推荐
- [获得最佳训练结果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)☘️
  推荐的
- [多 GPU 训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch 中心](https://github.com/ultralytics/yolov5/issues/36)🌟 新
- [TFLite、ONNX、CoreML、TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251)🚀
- [NVIDIA Jetson Nano 部署](https://github.com/ultralytics/yolov5/issues/9627)🌟 新
- [测试时间增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型集成](https://github.com/ultralytics/yolov5/issues/318)
- [模型修剪/稀疏度](https://github.com/ultralytics/yolov5/issues/304)
- [超参数进化](https://github.com/ultralytics/yolov5/issues/607)
- [使用冻结层进行迁移学习](https://github.com/ultralytics/yolov5/issues/1314)
- [架构总结](https://github.com/ultralytics/yolov5/issues/6998)🌟 新
- [用于数据集、标签和主动学习的 Roboflow](https://github.com/ultralytics/yolov5/issues/4975)🌟 新
- [ClearML 记录](https://github.com/ultralytics/yolov5/tree/master/utils/loggers/clearml)🌟 新
- [所以平台](https://github.com/ultralytics/yolov5/wiki/Deci-Platform)🌟 新
- [彗星记录](https://github.com/ultralytics/yolov5/tree/master/utils/loggers/comet)🌟 新

</details>

## <div align="center">集成</div>

<br>
<a align="center" href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/master/im/integrations-loop.png"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-clearml.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-readme-comet">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-comet.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-deci-platform">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-deci.png" width="10%" /></a>
</div>

|                                    机器人流                                     |                                ClearML ⭐ 新                                |                                      彗星⭐新                                      |                               所以⭐新                                |
| :-------------------------------------------------------------------------: | :-----------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :---------------------------------------------------------------: |
| 将您的自定义数据集标记并直接导出到 YOLOv5 以进行训练[机器人流](https://roboflow.com/?ref=ultralytics) | 使用自动跟踪、可视化甚至远程训练 YOLOv5[清除ML](https://cutt.ly/yolov5-readme-clearml)（开源！） | 永远免费，[彗星](https://bit.ly/yolov5-readme-comet)可让您保存 YOLOv5 模型、恢复训练以及交互式可视化和调试预测 | 一键自动编译量化YOLOv5以获得更好的推理性能[所以](https://bit.ly/yolov5-deci-platform) |

## <div align="center">Ultralytics 集线器</div>

[Ultralytics 集线器](https://bit.ly/ultralytics_hub)是我们的⭐**新的**用于可视化数据集、训练 YOLOv5 🚀 模型并以无缝体验部署到现实世界的无代码解决方案。开始使用**自由的**现在！

<a align="center" href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/master/im/ultralytics-hub.png"></a>

## <div align="center">为什么选择 YOLOv5</div>

YOLOv5 被设计为超级容易上手和简单易学。我们优先考虑现实世界的结果。

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP 值**表示[map@0.5](mailto:mAP@0.5):0.95 指标在 5000 张图像上测得[COCO val2017](http://cocodataset.org)从 256 到 1536 的各种推理大小的数据集。
- **显卡速度**测量每张图像的平均推理时间[COCO val2017](http://cocodataset.org)数据集使用[美国销售.Excelerge](https://aws.amazon.com/ec2/instance-types/p3/)批量大小为 32 的 V100 实例。
- **高效**数据来自[谷歌/汽车](https://github.com/google/automl)批量大小为 8。
- **复制**经过`python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### 预训练检查点

| 模型                                                                                                  | 尺寸<br><sup>（像素） | 地图<sup>值<br>50-95 | 地图<sup>值<br>50   | 速度<br><sup>处理器b1<br>（小姐） | 速度<br><sup>V100 b1<br>（小姐） | 速度<br><sup>V100 b32<br>（小姐） | 参数<br><sup>(男) | 失败者<br><sup>@640（二） |
| --------------------------------------------------------------------------------------------------- | --------------- | ----------------- | ---------------- | ------------------------ | -------------------------- | --------------------------- | -------------- | ------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt)                  | 640             | 28.0              | 45.7             | **45**                   | **6.3**                    | **0.6**                     | **1.9**        | **4.5**             |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt)                  | 640             | 37.4              | 56.8             | 98                       | 6.4                        | 0.9                         | 7.2            | 16.5                |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt)                  | 640             | 45.4              | 64.1             | 224                      | 8.2                        | 1.7                         | 21.2           | 49.0                |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt)                  | 640             | 49.0              | 67.3             | 430                      | 10.1                       | 2.7                         | 46.5           | 109.1               |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt)                  | 640             | 50.7              | 68.9             | 766                      | 12.1                       | 4.8                         | 86.7           | 205.7               |
|                                                                                                     |                 |                   |                  |                          |                            |                             |                |                     |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n6.pt)                | 1280            | 36.0              | 54.4             | 153                      | 8.1                        | 2.1                         | 3.2            | 4.6                 |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s6.pt)                | 1280            | 44.8              | 63.7             | 385                      | 8.2                        | 3.6                         | 12.6           | 16.8                |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m6.pt)                | 1280            | 51.3              | 69.3             | 887                      | 11.1                       | 6.8                         | 35.7           | 50.0                |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l6.pt)                | 1280            | 53.7              | 71.3             | 1784                     | 15.8                       | 10.5                        | 76.8           | 111.4               |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt)<br>+[电讯局][tta] | 1280<br>1536    | 55.0<br>**55.8**  | 72.7<br>**72.7** | 3136<br>-                | 26.2<br>-                  | 19.4<br>-                   | 140.7<br>-     | 209.8<br>-          |

<details>
  <summary>Table Notes</summary>

- 所有检查点都使用默认设置训练到 300 个时期。纳米和小型型号使用[hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml)hyps，所有其他人都使用[hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- \*\*地图<sup>值</sup>\*\*值适用于单模型单尺度[COCO val2017](http://cocodataset.org)数据集。<br>重现者`python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **速度**使用 a 对 COCO val 图像进行平均[美国销售.Excelerge](https://aws.amazon.com/ec2/instance-types/p3/)实例。 NMS 时间 (~1 ms/img) 不包括在内。<br>重现者`python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **电讯局**[测试时间增加](https://github.com/ultralytics/yolov5/issues/303)包括反射和尺度增强。<br>重现者`python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">分类⭐新</div>

YOLOv5[发布 v6.2](https://github.com/ultralytics/yolov5/releases)带来对分类模型训练、验证和部署的支持！查看我们的完整详细信息[发行说明](https://github.com/ultralytics/yolov5/releases/v6.2)并访问我们的[YOLOv5 分类 Colab 笔记本](https://github.com/ultralytics/yolov5/blob/master/classify/tutorial.ipynb)快速入门教程。

<details>
  <summary>Classification Checkpoints</summary>

<br>

我们使用 4xA100 实例在 ImageNet 上训练了 90 个时期的 YOLOv5-cls 分类模型，我们训练了 ResNet 和 EfficientNet 模型以及相同的默认训练设置以进行比较。我们将所有模型导出到 ONNX FP32 以进行 CPU 速度测试，并导出到 TensorRT FP16 以进行 GPU 速度测试。我们在 Google 上进行了所有速度测试[协作临](https://colab.research.google.com/signup)为了便于重现。

| 模型                                                                                         | 尺寸<br><sup>（像素） | acc<br><sup>top1 | acc<br><sup>烹饪 | 训练<br><sup>90个纪元<br>4xA100（小时） | 速度<br><sup>ONNX 中央处理器<br>（小姐） | 速度<br><sup>TensorRT V100<br>（小姐） | 参数<br><sup>(男) | 失败者<br><sup>@224（二） |
| ------------------------------------------------------------------------------------------ | --------------- | ---------------- | -------------- | ------------------------------ | ----------------------------- | -------------------------------- | -------------- | ------------------- |
| [YOLOv5n-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-cls.pt) | 224             | 64.6             | 85.4           | 7:59                           | **3.3**                       | **0.5**                          | **2.5**        | **0.5**             |
| [YOLOv5s-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s-cls.pt) | 224             | 71.5             | 90.2           | 8:09                           | 6.6                           | 0.6                              | 5.4            | 1.4                 |
| [YOLOv5m-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m-cls.pt) | 224             | 75.9             | 92.9           | 10:06                          | 15.5                          | 0.9                              | 12.9           | 3.9                 |
| [YOLOv5l-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt) | 224             | 78.0             | 94.0           | 11:56                          | 26.9                          | 1.4                              | 26.5           | 8.5                 |
| [YOLOv5x-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x-cls.pt) | 224             | **79.0**         | **94.4**       | 15:04                          | 54.3                          | 1.8                              | 48.1           | 15.9                |
|                                                                                            |                 |                  |                |                                |                               |                                  |                |                     |
| [ResNet18](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet18.pt)       | 224             | 70.3             | 89.5           | **6:47**                       | 11.2                          | 0.5                              | 11.7           | 3.7                 |
| [Resnetzch](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet34.pt)      | 224             | 73.9             | 91.8           | 8:33                           | 20.6                          | 0.9                              | 21.8           | 7.4                 |
| [ResNet50](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet50.pt)       | 224             | 76.8             | 93.4           | 11:10                          | 23.4                          | 1.0                              | 25.6           | 8.5                 |
| [ResNet101](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet101.pt)     | 224             | 78.5             | 94.3           | 17:10                          | 42.1                          | 1.9                              | 44.5           | 15.9                |
|                                                                                            |                 |                  |                |                                |                               |                                  |                |                     |
| [高效网络_b0](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b0.pt) | 224             | 75.1             | 92.4           | 13:03                          | 12.5                          | 1.3                              | 5.3            | 1.0                 |
| [高效网络 b1](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b1.pt) | 224             | 76.4             | 93.2           | 17:04                          | 14.9                          | 1.6                              | 7.8            | 1.5                 |
| [我们将预测](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b2.pt)   | 224             | 76.6             | 93.4           | 17:10                          | 15.9                          | 1.6                              | 9.1            | 1.7                 |
| [高效Netb3](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b3.pt) | 224             | 77.7             | 94.0           | 19:19                          | 18.9                          | 1.9                              | 12.2           | 2.4                 |

<details>
  <summary>Table Notes (click to expand)</summary>

- 使用 SGD 优化器将所有检查点训练到 90 个时期`lr0=0.001`和`weight_decay=5e-5`在图像大小 224 和所有默认设置。<br>运行记录到[HTTPS://玩豆瓣.爱/Glenn-就ocher/yo lo V5-classifier-V6-2](https://wandb.ai/glenn-jocher/YOLOv5-Classifier-v6-2)
- **准确性**值适用于单模型单尺度[ImageNet-1k](https://www.image-net.org/index.php)数据集。<br>重现者`python classify/val.py --data ../datasets/imagenet --img 224`
- **速度**使用谷歌平均超过 100 个推理图像[协作临](https://colab.research.google.com/signup)V100 高 RAM 实例。<br>重现者`python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`
- **出口**到 FP32 的 ONNX 和 FP16 的 TensorRT 完成`export.py`.<br>重现者`python export.py --weights yolov5s-cls.pt --include engine onnx --imgsz 224`
  </details>
  </details>

<details>
  <summary>Classification Usage Examples &nbsp;<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/classify/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></summary>

### 火车

YOLOv5 分类训练支持自动下载 MNIST、Fashion-MNIST、CIFAR10、CIFAR100、Imagenette、Imagewoof 和 ImageNet 数据集`--data`争论。开始使用 MNIST 进行训练`--data mnist`.

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### 瓦尔

在 ImageNet-1k 数据集上验证 YOLOv5m-cls 的准确性：

```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### 预测

使用预训练的 YOLOv5s-cls.pt 来预测 bus.jpg：

```bash
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```

```python
model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5s-cls.pt"
)  # load from PyTorch Hub
```

### 出口

将一组经过训练的 YOLOv5s-cls、ResNet 和 EfficientNet 模型导出到 ONNX 和 TensorRT：

```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```

</details>

## <div align="center">环境</div>

在几秒钟内开始使用我们经过验证的环境。单击下面的每个图标了解详细信息。

<div align="center">
  <a href="https://bit.ly/yolov5-paperspace-notebook">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gradient.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://www.kaggle.com/ultralytics/yolov5">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://hub.docker.com/r/ultralytics/yolov5">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="10%" /></a>
</div>

## <div align="center">贡献</div>

我们喜欢您的意见！我们希望尽可能简单和透明地为 YOLOv5 做出贡献。请看我们的[投稿指南](CONTRIBUTING.md)开始，并填写[YOLOv5调查](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)向我们发送您的体验反馈。感谢我们所有的贡献者！

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors"><img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/image-contributors-1280.png" /></a>

## <div align="center">执照</div>

YOLOv5 在两种不同的许可下可用：

- **GPL-3.0 许可证**： 看[执照](https://github.com/ultralytics/yolov5/blob/master/LICENSE)文件的详细信息。
- **企业执照**：在没有 GPL-3.0 开源要求的情况下为商业产品开发提供更大的灵活性。典型用例是将 Ultralytics 软件和 AI 模型嵌入到商业产品和应用程序中。在以下位置申请企业许可证[Ultralytics 许可](https://ultralytics.com/license).

## <div align="center">接触</div>

对于 YOLOv5 错误和功能请求，请访问[GitHub 问题](https://github.com/ultralytics/yolov5/issues).如需专业支持，请[联系我们](https://ultralytics.com/contact).

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

[tta]: https://github.com/ultralytics/yolov5/issues/303
