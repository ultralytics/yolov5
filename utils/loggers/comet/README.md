<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

<img src="https://cdn.comet.ml/img/notebook_logo.png">

# Using Ultralytics YOLO With Comet

Welcome to the guide for integrating [Ultralytics YOLO](https://github.com/ultralytics/yolov5) with [Comet](https://www.comet.com/site/)! Comet offers robust experiment tracking, model management, and visualization tools to enhance your [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workflow. This guide explains how to leverage Comet for monitoring training, logging results, managing datasets, and optimizing hyperparameters for your YOLO models.

[![Ultralytics Actions](https://github.com/ultralytics/velocity/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/velocity/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🧪 About Comet

[Comet](https://www.comet.com/site/) provides tools for data scientists, engineers, and teams to accelerate and optimize [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) and machine learning models.

With Comet, you can track and visualize model metrics in real time, save [hyperparameters](https://docs.ultralytics.com/guides/hyperparameter-tuning/), datasets, and model checkpoints, and visualize predictions using Custom Panels. Comet ensures you never lose track of your work and makes sharing results and collaborating across teams seamless. For more details, see the [Comet Documentation](https://www.comet.com/docs/v2/).

## 🚀 Getting Started

Follow these steps to set up Comet for your YOLO projects.

### Install Comet

Install the [comet_ml Python package](https://pypi.org/project/comet-ml/) using pip:

```shell
pip install comet_ml
```

### Configure Comet Credentials

You can configure Comet in two ways:

1. **Environment Variables:**  
   Set your credentials directly in your environment.

   ```shell
   export COMET_API_KEY=YOUR_COMET_API_KEY
   export COMET_PROJECT_NAME=YOUR_COMET_PROJECT_NAME # Defaults to 'yolov5' if not set
   ```

   Find your API key in your [Comet Account Settings](https://www.comet.com/site/).

2. **Configuration File:**  
   Create a `.comet.config` file in your working directory:

   ```ini
   [comet]
   api_key=YOUR_COMET_API_KEY
   project_name=YOUR_COMET_PROJECT_NAME # Defaults to 'yolov5' if not set
   ```

### Run the Training Script

Run the YOLO [training script](https://docs.ultralytics.com/modes/train/). Comet will automatically log your run.

```shell
# Train YOLO on COCO128 for 5 epochs
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```

Comet automatically logs hyperparameters, command-line arguments, and training/validation metrics. Visualize and analyze your runs in the Comet UI. For more details, see the [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/).

<img width="1920" alt="Comet UI showing YOLO training metrics" src="https://user-images.githubusercontent.com/26833433/202851203-164e94e1-2238-46dd-91f8-de020e9d6b41.png">

## ✨ Try an Example!

Explore a completed YOLO training run tracked with Comet:

- **[View Example Run on Comet](https://www.comet.com/examples/comet-example-yolov5/a0e29e0e9b984e4a822db2a62d0cb357?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github_readme)**

Run the example yourself using this [Google Colab](https://colab.research.google.com/) notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/model-training/yolov5/notebooks/Comet_and_YOLOv5.ipynb)

## 📊 Automatic Logging

Comet automatically logs the following information by default:

### Metrics

- **Losses:** Box Loss, Object Loss, Classification Loss (Training and Validation)
- **Performance:** [mAP@0.5](https://www.ultralytics.com/glossary/mean-average-precision-map), mAP@0.5:0.95 (Validation). Learn more in the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
- **[Precision](https://www.ultralytics.com/glossary/precision) and [Recall](https://www.ultralytics.com/glossary/recall):** Validation data metrics

### Parameters

- **Model Hyperparameters:** Configuration used for the model
- **Command Line Arguments:** All arguments passed via the [CLI](https://docs.ultralytics.com/usage/cli/)

### Visualizations

- **[Confusion Matrix](https://www.ultralytics.com/glossary/confusion-matrix):** Model predictions on validation data ([Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix))
- **Curves:** PR and F1 curves across all classes
- **Label Correlogram:** Correlation visualization of class labels

## ⚙️ Advanced Configuration

Customize Comet's logging behavior using command-line flags or environment variables.

```shell
# Environment Variables for Comet Configuration
export COMET_MODE=online                                    # 'online' or 'offline'. Default: online
export COMET_MODEL_NAME=YOUR_MODEL_NAME                     # Name for the saved model. Default: yolov5
export COMET_LOG_CONFUSION_MATRIX=false                     # Disable confusion matrix logging. Default: true
export COMET_MAX_IMAGE_UPLOADS=NUMBER                       # Max prediction images to log. Default: 100
export COMET_LOG_PER_CLASS_METRICS=true                     # Log metrics per class. Default: false
export COMET_DEFAULT_CHECKPOINT_FILENAME=checkpoint_file.pt # Checkpoint for resuming. Default: 'last.pt'
export COMET_LOG_BATCH_LEVEL_METRICS=true                   # Log training metrics per batch. Default: false
export COMET_LOG_PREDICTIONS=true                           # Disable prediction logging if set to false. Default: true
```

For more configuration options, see the [Comet documentation](https://www.comet.com/docs/v2/).

### Logging Checkpoints With Comet

Model checkpoint logging to Comet is disabled by default. Enable it using the `--save-period` argument during training to save checkpoints at the specified epoch interval.

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --save-period 1 # Save checkpoint every epoch
```

Checkpoints will appear in the "Assets & Artifacts" tab of your Comet experiment. Learn more about model management in the [Comet Model Registry documentation](https://www.comet.com/docs/v2/guides/model-registry/using-model-registry/).

### Logging Model Predictions

By default, model predictions (images, ground truth labels, [bounding boxes](https://www.ultralytics.com/glossary/bounding-box)) for the validation set are logged. Control the logging frequency using the `--bbox_interval` argument, which specifies logging every Nth batch per epoch.

**Note:** The YOLO validation dataloader defaults to a batch size of 32. Adjust `--bbox_interval` as needed.

Visualize predictions using Comet's Object Detection Custom Panel. See an [example project using the Panel](https://www.comet.com/examples/comet-example-yolov5?shareable=YcwMiJaZSXfcEXpGOHDD12vA1&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github_readme).

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --bbox_interval 2 # Log predictions every 2nd validation batch per epoch
```

#### Controlling the Number of Prediction Images

Adjust the maximum number of validation images logged using the `COMET_MAX_IMAGE_UPLOADS` environment variable.

```shell
env COMET_MAX_IMAGE_UPLOADS=200 python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --bbox_interval 1 # Log every batch
```

### Logging Class Level Metrics

Enable logging of mAP, precision, recall, and F1-score for each class using the `COMET_LOG_PER_CLASS_METRICS` environment variable.

```shell
env COMET_LOG_PER_CLASS_METRICS=true python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt
```

## 💾 Dataset Management With Comet Artifacts

Use [Comet Artifacts](https://www.comet.com/docs/v2/guides/artifacts/using-artifacts/) to version, store, and manage your datasets.

### Uploading a Dataset

Upload your dataset using the `--upload_dataset` flag. Ensure your dataset follows the structure described in the [Ultralytics Datasets documentation](https://docs.ultralytics.com/datasets/) and that your dataset config [YAML](https://www.ultralytics.com/glossary/yaml) file matches the format of `coco128.yaml` (see the [COCO128 dataset docs](https://docs.ultralytics.com/datasets/detect/coco128/)).

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --upload_dataset # Upload the dataset specified in coco128.yaml
```

View the uploaded dataset in the Artifacts tab of your Comet Workspace.
<img width="1073" alt="Comet Artifacts tab showing uploaded dataset" src="https://user-images.githubusercontent.com/7529846/186929193-162718bf-ec7b-4eb9-8c3b-86b3763ef8ea.png">

Preview data directly in the Comet UI.
<img width="1082" alt="Comet UI previewing dataset images" src="https://user-images.githubusercontent.com/7529846/186929215-432c36a9-c109-4eb0-944b-84c2786590d6.png">

Artifacts are versioned and support metadata. Comet automatically logs metadata from your dataset YAML file.
<img width="963" alt="Comet Artifact metadata view" src="https://user-images.githubusercontent.com/7529846/186929256-9d44d6eb-1a19-42de-889a-bcbca3018f2e.png">

### Using a Saved Artifact

To use a dataset stored in Comet Artifacts, update the `path` in your dataset YAML file to the Artifact resource URL:

```yaml
# contents of artifact.yaml
path: "comet://WORKSPACE_NAME/ARTIFACT_NAME:ARTIFACT_VERSION_OR_ALIAS"
train: images/train # Adjust subdirectory if needed
val: images/val # Adjust subdirectory if needed

# Other dataset configurations...
```

Then, pass this configuration file to your training script:

```shell
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data artifact.yaml \
  --weights yolov5s.pt
```

Artifacts track data lineage, showing which experiments used specific dataset versions.
<img width="1391" alt="Comet Artifact lineage graph" src="https://user-images.githubusercontent.com/7529846/186929264-4c4014fa-fe51-4f3c-a5c5-f6d24649b1b4.png">

## 🔄 Resuming Training Runs

If a training run is interrupted (for example, due to connection issues), you can resume it using the `--resume` flag with the Comet Run Path (`comet://YOUR_WORKSPACE/YOUR_PROJECT/EXPERIMENT_ID`).

This restores the model state, hyperparameters, arguments, and downloads necessary Artifacts, continuing logging to the existing Comet Experiment. Learn more about [resuming runs in the Comet documentation](https://www.comet.com/docs/v2/guides/experiment-management/resume-experiment/).

```shell
python train.py \
  --resume "comet://YOUR_WORKSPACE/YOUR_PROJECT/EXPERIMENT_ID"
```

## 🔍 Hyperparameter Optimization (HPO)

YOLO integrates with the [Comet Optimizer](https://www.comet.com/docs/v2/guides/optimizer/configure-optimizer/) for easy hyperparameter sweeps and visualization. This helps you find the best set of parameters for your model, a process often referred to as [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

### Configuring an Optimizer Sweep

Create a [JSON](https://www.ultralytics.com/glossary/json) configuration file defining the sweep parameters, search strategy, and objective metric. An example is provided at `utils/loggers/comet/optimizer_config.json`.

Run the sweep using the `hpo.py` script:

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json"
```

The `hpo.py` script accepts the same arguments as `train.py`. Pass additional fixed arguments for the sweep:

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json" \
  --save-period 1 \
  --bbox_interval 1
```

### Running a Sweep in Parallel

Execute multiple sweep trials concurrently using the `comet optimizer` command:

```shell
comet optimizer -j \
  utils/loggers/comet/hpo.py NUM_WORKERS utils/loggers/comet/optimizer_config.json
```

Replace `NUM_WORKERS` with the desired number of parallel processes.

### Visualizing HPO Results

Comet offers various visualizations for analyzing sweep results, such as parallel coordinate plots and parameter importance plots. Explore a [project with a completed sweep](https://www.comet.com/examples/comet-example-yolov5/view/PrlArHGuuhDTKC1UuBmTtOSXD/panels?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github_readme).

<img width="1626" alt="Comet HPO visualization" src="https://user-images.githubusercontent.com/7529846/186914869-7dc1de14-583f-4323-967b-c9a66a29e495.png">

## 🤝 Contributing

Contributions to enhance the YOLO-Comet integration are welcome! Please see the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more information on how to get involved. Thank you for helping improve this integration!
