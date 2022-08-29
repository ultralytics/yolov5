<img src="https://cdn.comet.ml/img/notebook_logo.png">

# YOLOv5 with Comet

This guide will cover how to use YOLOv5 with [Comet](https://www.comet.com/site/?ref=yolov5)

# About Comet

Comet builds tools that help data scientists, engineers, and team leaders accelerate and optimize machine learning and deep learning models.

Track and visualize model metrics in real time, save your hyperparameters, datasets, and model checkpoints, and visualize your model predictions with [Comet Custom Panels](https://www.comet.com/examples/comet-example-yolov5/view/1c4Dqcu8mZ767NBipjwlx3gz6/panels?ref=yolov5)!

Comet makes sure you never lose track of your work and makes it easy to share results and collaborate across teams of all sizes!

# Getting Started

## Install Comet

```shell
pip install comet_ml
```

## Configure Comet Credentials

There are two ways to configure Comet with YOLOv5.

You can either set your credentials through enviroment variables

**Environment Variables**

```shell
export COMET_API_KEY=<Your Comet API Key>
export COMET_PROJECT_NAME=<Your Comet Project Name> # This will default to 'yolov5'
```

Or create a `.comet.config` file in your working directory and set your credentials there.

**Comet Configuration File**

```
[comet]
api_key=<Your Comet API Key>
project_name=<Your Comet Project Name> # This will default to 'yolov5'
```

## Run the Training Script

```shell
# Train YOLOv5s on COCO128 for 5 epochs
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```

That's it! Comet will automatically log your hyperparameters, command line arguments, training and valiation metrics. You can visualize and analyze your runs in the Comet UI

<img width="800" alt="yolo-ui" src="https://user-images.githubusercontent.com/7529846/186725929-bc18019f-cec9-45b9-978e-496f4a628ab7.png">

# Try out an Example!
Check out an example of a [completed run here](https://www.comet.com/examples/comet-example-yolov5/353f9734261348b59b883660bcd62256?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step&ref=yolov5)

Or better yet, try it out yourself in this Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

# Log automatically

By default, Comet will log the following items

## Metrics
- Box Loss, Object Loss, Classification Loss for the training and validation data
- mAP_0.5, mAP_0.5:0.95 metrics for the validation data.
- Precision and Recall for the validation data

## Parameters

- Model Hyperparameters
- All parameters passed through the command line options

## Visualizations

- Confusion Matrix of the model predictions on the validation data
- Plots for the PR and F1 curves across all classes
- Correlogram of the Class Labels

# Configure Comet Logging

Comet can be configured to log additional data either through command line flags passed to the training script
or through environment variables.

Here is the full list of command line options for Comet

```shell
  --comet_mode COMET_MODE
                        Comet: Set whether to run Comet in online
                        or offline mode.
  --comet_save_model    Comet: Set to save model checkpoints.
  --comet_model_name COMET_MODEL_NAME
                        Comet: Set the name for the saved model.
  --comet_overwrite_checkpoints
                        Comet: Overwrite exsiting model
                        checkpoints.
  --comet_checkpoint_filename [COMET_CHECKPOINT_FILENAME]
                        Comet: Name of the checkpoint file to save
                        to Comet.Set to 'all' to log all
                        checkpoints.
  --comet_checkpoint_step COMET_CHECKPOINT_STEP
  --comet_log_batch_metrics
                        Comet: Set to log batch level training
                        metrics.
  --comet_log_batch_interval COMET_LOG_BATCH_INTERVAL
                        Comet: Logging frequency for batch level
                        training metrics.
  --comet_log_prediction_interval COMET_LOG_PREDICTION_INTERVAL
                        Comet: How often to log predictions.Applied
                        at batch level.
  --comet_log_confusion_matrix
                        Comet: Log a Confusion Matrix for the
                        validation dataset.
  --comet_log_predictions
                        Comet: Log Predictions on Images from the
                        Validation Set
  --comet_max_image_uploads COMET_MAX_IMAGE_UPLOADS
                        Comet: Maximum number of images to log to
                        Comet.
  --comet_upload_dataset [COMET_UPLOAD_DATASET]
                        Comet: Upload Dataset to Comet as an
                        Artifact.Set to 'train', 'val' or 'test' to
                        upload a single dataset.
  --comet_artifact COMET_ARTIFACT
                        Comet: Name of the Comet dataset Artifact
                        to download.
```

Let's take a look at these options.

## Logging Checkpoints with Comet

Logging Models to Comet is disabled by default. To enable it, pass the `comet_save_model` flag to the training script.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--save-period 1 \
--comet_save_model
```

You have a few options when it comes to saving model checkpoints to Comet. Here's how you can change the checkpointing behavior.

### Change which checkpoint file is logged
By default, Comet will save the `best.pt` checkpoint. To change which file gets saved, use the `comet_checkpoint_filename` argument.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--save-period 1 \
--comet_save_model \
--comet_checkpoint_filename "last.pt" # this defaults to "best.pt"
```

### Log all checkpoints files

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--save-period 1 \
--comet_save_model \
--comet_checkpoint_filename "all"
```

### Overwrite checkpoint files

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--save-period 1 \
--comet_save_model \
--comet_checkpoint_filename "best.pt" \
--comet_overwrite_checkpoint
```

## Using a saved Checkpoint

Comet will log a Run Path for every run that you can use to download model weights and resume runs. A run path is a string with the following format `comet://<your workspace name>/<your project name>/<experiment id>`

You can find the run path in the `Others` tab in your Comet Experiment.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights "comet://examples/comet-example-yolov5/353f9734261348b59b883660bcd62256" \
```

By default, Comet will download the most recent checkpoint, this can be configured by specifying the checkpoint filename. You can specify a checkpoint for a particular epoch
by using the `comet_checkpoint_filename` flag.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights "comet://examples/comet-example-yolov5/353f9734261348b59b883660bcd62256" \
--comet_checkpoint_filename "last_epoch_2.pt"
```

## Logging Model Predictions

You can log model predictions and the associated images using `comet_log_predictions`. Predictions can be visualized using Comet's Object Detection Custom Panel

Here is an [example project using the Panel](https://www.comet.com/examples/comet-example-yolov5/view/1c4Dqcu8mZ767NBipjwlx3gz6/panels?ref=yolov5)

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--comet_log_predictions
```

### Controlling the number of Prediction Images logged to Comet

When logging predictions from YOLOv5, Comet will log the images associated with each set of predictions. By default a maximum of 100 validation images are logged. You can increase or decrease this number using the `comet_max_images` flag.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--comet_log_predictions \
--comet_max_images 200
```

### Controlling the frequency of Prediction Images logged to Comet

By default, every batch from the validation set is logged to Comet when logging predictions. This can be configured via the `comet_log_prediction_interval` flag. For example, setting this parameter to `5` will log every 5th batch of data from the validation set.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--comet_log_predictions \
--comet_log_prediction_interval 5
```

## Uploading a Dataset to Comet Artifacts

If you would like to store your data using [Comet Artifacts](https://www.comet.com/docs/v2/guides/data-management/using-artifacts/#learn-more?ref=yolov5), you can do so using the `comet_upload_dataset` flag.

The dataset be organized in the way described in the [YOLOv5 documentation](https://docs.ultralytics.com/tutorials/train-custom-datasets/#3-organize-directories). The dataset config `yaml` file must follow the same format as that of the `coco128.yaml` file.

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--comet_upload_dataset
```
You can find the uploaded dataset in the Artifacts tab in your Comet Workspace
<img width="1073" alt="artifact-1" src="https://user-images.githubusercontent.com/7529846/186929193-162718bf-ec7b-4eb9-8c3b-86b3763ef8ea.png">

You can preview the data directly in the Comet UI.
<img width="1082" alt="artifact-2" src="https://user-images.githubusercontent.com/7529846/186929215-432c36a9-c109-4eb0-944b-84c2786590d6.png">

Artifacts are versioned and also support adding metadata about the dataset. Comet will automatically log the metadata from your dataset `yaml` file
<img width="963" alt="artifact-3" src="https://user-images.githubusercontent.com/7529846/186929256-9d44d6eb-1a19-42de-889a-bcbca3018f2e.png">

### Using a saved Artifact

If you would like to use a dataset from Comet Artifacts, simply pass the `comet_artifact` flag to the training script, along with the artifact path. The artifact path is a string with the following format `<workspace name>/<artifact name>:<artifact version or alias>`

```shell
python train.py \
--img 640 \
--batch 16 \
--epochs 5 \
--data coco128.yaml \
--weights yolov5s.pt \
--comet_artifact "examples/yolov5-dataset:latest"
```

Artifacts also allow you to track the lineage of data as it flows through your Experimentation workflow. Here you can see a graph that shows you all the experiments that have used your uploaded dataset.
<img width="1391" alt="artifact-4" src="https://user-images.githubusercontent.com/7529846/186929264-4c4014fa-fe51-4f3c-a5c5-f6d24649b1b4.png">

## Resuming a Training Run

If your training run is interrupted for any reason, e.g. disrupted internet connection, you can resume the run using the `resume` flag and the Comet Run Path.

The Run Path has the following format `comet://<your workspace name>/<your project name>/<experiment id>`.

This will restore the run to its state before the interruption, which includes restoring the  model from a checkpoint, restoring all hyperparameters and training arguments and downloading Comet dataset Artifacts if they were used in the original run. The resumed run will continue logging to the existing Experiment in the Comet UI

```shell
python train.py \
--resume "comet://examples/comet-example-yolov5/353f9734261348b59b883660bcd62256"
```

## Hyperparameter Search with the Comet Optimizer

YOLOv5 is also integrated with Comet's Optimizer, making is simple to visualie hyperparameter sweeps in the Comet UI.

### Configuring an Optimizer Sweep

To configure the Comet Optimizer, you will have to create a JSON file with the information about the sweep. An example file has been provided [here](optimizer_config.json)

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json" \
```

The `hpo.py` script accepts the same arguments as `train.py`. If you wish to pass additional Comet related arguments to your sweep simply add them after

```shell
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json" \
  --comet_save_model \
  --comet_overwrite_checkpoints
```

### Running a Sweep in Parallel

```shell
comet optimizer -j <set number of workers> utils/loggers/comet/hpo.py \
  utils/loggers/comet/optimizer_config.json"
```

The `hpo.py` script accepts the same arguments as `train.py`. If you wish to pass additional Comet related arguments to your sweep simply add them after

```shell
comet optimizer -j <set number of workers> utils/loggers/comet/hpo.py \
  utils/loggers/comet/optimizer_config.json" \
  --comet_save_model \
  --comet_overwrite_checkpoints
```

### Visualizing Results

Comet provides a number of ways to visualize the results of your sweep. Take a look at a [project with a completed sweep here](https://www.comet.com/examples/comet-example-yolov5/view/PrlArHGuuhDTKC1UuBmTtOSXD/panels?ref=yolov5)

<img width="1626" alt="hyperparameter-yolo" src="https://user-images.githubusercontent.com/7529846/186914869-7dc1de14-583f-4323-967b-c9a66a29e495.png">