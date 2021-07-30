📚 This guide explains how to use **Weights & Biases** (W&B) with YOLOv5 🚀.
 * [About Weights & Biases](#about-weights-&-biases)
 * [First-Time Setup](#first-time-setup)
 * [Viewing runs](#viewing-runs)
 * [Advanced Usage: Dasaset and Model Management](#advanced-usage)
 * [Reports: Share your work with the world!](#reports)
 
## About Weights & Biases
Think of [W&B](https://wandb.ai/site?utm_campaign=repo_yolo_wandbtutorial) like GitHub for machine learning models. With a few lines of code, save everything you need to debug, compare and reproduce your models — architecture, hyperparameters, git commits, model weights, GPU usage, and even datasets and predictions.
  
 Used by top researchers including teams at OpenAI, Lyft, Github, and MILA, W&B is part of the new standard of best practices for machine learning. How W&B can help you optimize your machine learning workflows:
 
 * [Debug](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Free-2) model performance in real time
 * [GPU usage](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#System-4), visualized automatically
 * [Custom charts](https://wandb.ai/wandb/customizable-charts/reports/Powerful-Custom-Charts-To-Debug-Model-Peformance--VmlldzoyNzY4ODI) for powerful, extensible visualization
 * [Share insights](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Share-8) interactively with collaborators
 * [Optimize hyperparameters](https://docs.wandb.com/sweeps) efficiently
 * [Track](https://docs.wandb.com/artifacts) datasets, pipelines, and production models
 
<details open>
<summary>## First-Time Setup</summary>
When you first train, W&B will prompt you to create a new account and will generate an **API key** for you. If you are an existing user you can retrieve your key from https://wandb.ai/authorize. This key is used to tell W&B where to log your data. You only need to supply your key once, and then it is remembered on the same device.
 
 W&B will create a cloud **project** (default is 'YOLOv5') for your training runs, and each new training run will be provided a unique run **name** within that project as project/name. You can also manually set your project and run name as:
 
 ```shell
 $ python train.py --project ... --name ...
 ```
 
 <img alt="" width="800" src="https://user-images.githubusercontent.com/26833433/98183367-4acbc600-1f08-11eb-9a23-7266a4192355.jpg">
 </details>
 
<details open>
<summary> ## Viewing Runs </summary>
 Run information streams from your environment to the W&B cloud console as you train. This allows you to monitor and even cancel runs in **realtime**. All important information is logged:
 
 * Training & Validation losses
 * Metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95
 * Learning Rate over time
 * A bounding box debugging panel, showing the training progress over time
 * GPU: Type, **GPU Utilization**, power, temperature, **CUDA memory usage**
 * System: Disk I/0, CPU utilization, RAM memory usage
 * Your trained model as W&B Artifact
 * Environment: OS and Python types, Git repository and state, **training command**
 
 <img alt="" width="800" src="https://user-images.githubusercontent.com/26833433/98184457-bd3da580-1f0a-11eb-8461-95d908a71893.jpg">
</details>
<details>
<summary> ## Advanced Usage: Dasaset and Model Management</summary>

</details>
 ## Reports
 W&B Reports can be created from your saved runs for sharing online. Once a report is created you will receive a link you can use to publically share your results. Here is an example report created from the COCO128 tutorial trainings of all four YOLOv5 models ([link](https://wandb.ai/glenn-jocher/yolov5_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY)).
 
 <img alt="" width="800" src="https://user-images.githubusercontent.com/26833433/98185222-794ba000-1f0c-11eb-850f-3e9c45ad6949.jpg">
 
 ## Environments
 YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):
 
 * **Google Colab and Kaggle** notebooks with free GPU: [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) [![Open In Kaggle](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/ultralytics/yolov5)
 * **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
 * **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
 * **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) [![Docker Pulls](https://camo.githubusercontent.com/280faedaf431e4c0c24fdb30ec00a66d627404e5c4c498210d3f014dd58c2c7e/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f756c7472616c79746963732f796f6c6f76353f6c6f676f3d646f636b6572)](https://hub.docker.com/r/ultralytics/yolov5)
 
 ## Status
 ![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)
 
 If this badge is green, all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are currently passing. CI tests verify correct operation of YOLOv5 training ([train.py](https://github.com/ultralytics/yolov5/blob/master/train.py)), validation ([val.py](https://github.com/ultralytics/yolov5/blob/master/val.py)), inference ([detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py)) and export ([export.py](https://github.com/ultralytics/yolov5/blob/master/export.py)) on MacOS, Windows, and Ubuntu every 24 hours and on every commit.

