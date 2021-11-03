📚 This guide explains how to use **Weights & Biases** (W&B) with YOLOv5 🚀. UPDATED 29 September 2021.
* [About Weights & Biases](#about-weights-&-biases)
* [First-Time Setup](#first-time-setup)
* [Viewing runs](#viewing-runs)
* [Advanced Usage: Dataset Versioning and Evaluation](#advanced-usage)
* [Reports: Share your work with the world!](#reports)

## About Weights & Biases
Think of [W&B](https://wandb.ai/site?utm_campaign=repo_yolo_wandbtutorial) like GitHub for machine learning models. With a few lines of code, save everything you need to debug, compare and reproduce your models — architecture, hyperparameters, git commits, model weights, GPU usage, and even datasets and predictions.

Used by top researchers including teams at OpenAI, Lyft, Github, and MILA, W&B is part of the new standard of best practices for machine learning. How W&B can help you optimize your machine learning workflows:

 * [Debug](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Free-2) model performance in real time
 * [GPU usage](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#System-4) visualized automatically
 * [Custom charts](https://wandb.ai/wandb/customizable-charts/reports/Powerful-Custom-Charts-To-Debug-Model-Peformance--VmlldzoyNzY4ODI) for powerful, extensible visualization
 * [Share insights](https://wandb.ai/wandb/getting-started/reports/Visualize-Debug-Machine-Learning-Models--VmlldzoyNzY5MDk#Share-8) interactively with collaborators
 * [Optimize hyperparameters](https://docs.wandb.com/sweeps) efficiently
 * [Track](https://docs.wandb.com/artifacts) datasets, pipelines, and production models

## First-Time Setup
<details open>
 <summary> Toggle Details </summary>
When you first train, W&B will prompt you to create a new account and will generate an **API key** for you. If you are an existing user you can retrieve your key from https://wandb.ai/authorize. This key is used to tell W&B where to log your data. You only need to supply your key once, and then it is remembered on the same device.

W&B will create a cloud **project** (default is 'YOLOv5') for your training runs, and each new training run will be provided a unique run **name** within that project as project/name. You can also manually set your project and run name as:

 ```shell
 $ python train.py --project ... --name ...
 ```

YOLOv5 notebook example: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<img width="960" alt="Screen Shot 2021-09-29 at 10 23 13 PM" src="https://user-images.githubusercontent.com/26833433/135392431-1ab7920a-c49d-450a-b0b0-0c86ec86100e.png">


 </details>

## Viewing Runs
<details open>
  <summary> Toggle Details </summary>
Run information streams from your environment to the W&B cloud console as you train. This allows you to monitor and even cancel runs in <b>realtime</b> . All important information is logged:

 * Training & Validation losses
 * Metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95
 * Learning Rate over time
 * A bounding box debugging panel, showing the training progress over time
 * GPU: Type, **GPU Utilization**, power, temperature, **CUDA memory usage**
 * System: Disk I/0, CPU utilization, RAM memory usage
 * Your trained model as W&B Artifact
 * Environment: OS and Python types, Git repository and state, **training command**

<p align="center"><img width="900" alt="Weights & Biases dashboard" src="https://user-images.githubusercontent.com/26833433/135390767-c28b050f-8455-4004-adb0-3b730386e2b2.png"></p>


</details>

## Advanced Usage
You can leverage W&B artifacts and Tables integration to easily visualize and manage your datasets, models and training evaluations. Here are some quick examples to get you started.
<details open>
 <h3>1. Visualize and Version Datasets</h3>
 Log, visualize, dynamically query, and understand your data with <a href='https://docs.wandb.ai/guides/data-vis/tables'>W&B Tables</a>. You can use the following command to log your dataset as a W&B Table. This will generate a <code>{dataset}_wandb.yaml</code> file which can be used to train from dataset artifact.
 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python utils/logger/wandb/log_dataset.py --project ... --name ... --data .. </code>

 ![Screenshot (64)](https://user-images.githubusercontent.com/15766192/128486078-d8433890-98a3-4d12-8986-b6c0e3fc64b9.png)
 </details>

 <h3> 2: Train and Log Evaluation simultaneousy </h3>
   This is an extension of the previous section, but it'll also training after uploading the dataset. <b> This also evaluation Table</b>
   Evaluation table compares your predictions and ground truths across the validation set for each epoch. It uses the references to the already uploaded datasets,
   so no images will be uploaded from your system more than once.
 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python utils/logger/wandb/log_dataset.py --data ..  --upload_data </code>

![Screenshot (72)](https://user-images.githubusercontent.com/15766192/128979739-4cf63aeb-a76f-483f-8861-1c0100b938a5.png)
 </details>

 <h3> 3: Train using dataset artifact </h3>
   When you upload a dataset as described in the first section, you get a new config file with an added `_wandb` to its name. This file contains the information that
   can be used to train a model directly from the dataset artifact. <b> This also logs evaluation </b>
 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python utils/logger/wandb/log_dataset.py --data {data}_wandb.yaml </code>

![Screenshot (72)](https://user-images.githubusercontent.com/15766192/128979739-4cf63aeb-a76f-483f-8861-1c0100b938a5.png)
 </details>

   <h3> 4: Save model checkpoints as artifacts </h3>
  To enable saving and versioning checkpoints of your experiment, pass `--save_period n` with the base cammand, where `n` represents checkpoint interval.
  You can also log both the dataset and model checkpoints simultaneously. If not passed, only the final model will be logged

 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python train.py --save_period 1 </code>

![Screenshot (68)](https://user-images.githubusercontent.com/15766192/128726138-ec6c1f60-639d-437d-b4ee-3acd9de47ef3.png)
 </details>

</details>

 <h3> 5: Resume runs from checkpoint artifacts. </h3>
Any run can be resumed using artifacts if the <code>--resume</code> argument starts with <code>wandb-artifact://</code> prefix followed by the run path, i.e, <code>wandb-artifact://username/project/runid </code>. This doesn't require the model checkpoint to be present on the local system.

 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python train.py --resume wandb-artifact://{run_path} </code>

![Screenshot (70)](https://user-images.githubusercontent.com/15766192/128728988-4e84b355-6c87-41ae-a591-14aecf45343e.png)
 </details>

  <h3> 6: Resume runs from dataset artifact & checkpoint artifacts. </h3>
 <b> Local dataset or model checkpoints are not required. This can be used to resume runs directly on a different device </b>
 The syntax is same as the previous section, but you'll need to lof both the dataset and model checkpoints as artifacts, i.e, set bot <code>--upload_dataset</code> or
 train from <code>_wandb.yaml</code> file and set <code>--save_period</code>

 <details>
  <summary> <b>Usage</b> </summary>
   <b>Code</b> <code> $ python train.py --resume wandb-artifact://{run_path} </code>

![Screenshot (70)](https://user-images.githubusercontent.com/15766192/128728988-4e84b355-6c87-41ae-a591-14aecf45343e.png)
 </details>

</details>


 <h3> Reports </h3>
W&B Reports can be created from your saved runs for sharing online. Once a report is created you will receive a link you can use to publically share your results. Here is an example report created from the COCO128 tutorial trainings of all four YOLOv5 models ([link](https://wandb.ai/glenn-jocher/yolov5_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY)).

<img width="900" alt="Weights & Biases Reports" src="https://user-images.githubusercontent.com/26833433/135394029-a17eaf86-c6c1-4b1d-bb80-b90e83aaffa7.png">


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab and Kaggle** notebooks with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>


## Status

![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)

If this badge is green, all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are currently passing. CI tests verify correct operation of YOLOv5 training ([train.py](https://github.com/ultralytics/yolov5/blob/master/train.py)), validation ([val.py](https://github.com/ultralytics/yolov5/blob/master/val.py)), inference ([detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py)) and export ([export.py](https://github.com/ultralytics/yolov5/blob/master/export.py)) on MacOS, Windows, and Ubuntu every 24 hours and on every commit.
