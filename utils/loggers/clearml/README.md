# ClearML Integration


## About ClearML
ClearML is an [open-source](https://github.com/allegroai/clearml) toolbox designed to save you time. It features (click on the arrow for screenshots):

<details closed>
<summary>🔨 An experiment manager</summary>
</details>
<details closed>
<summary>🔧 A data versioning tool</summary>
</details>
<details closed>
<summary>🔦 Remote execution using Queues and Workers</summary>
</details>
<details closed>
<summary>🔬 Hyperparameter Optimization</summary>
</details>
<details closed>
<summary>🔩 Pipelines</summary>
</details>
<details closed>
<summary>🔭 Serving</summary>
</details>

And so much more. It's up to you how many of these tools you want to use, you can stick to the experiment manager, or chain them all together into an impressive pipeline!

![ClearML scalars dashboard](https://github.com/thepycoder/clearml_screenshots/raw/main/experiment_manager.gif)


## 🦾 Setting things up
To keep track of your experiments and/or data, ClearML needs to communicate to a server. You have 2 options to get one:

1. Either sign up for free to the [ClearML Hosted Service](https://app.clear.ml) or you can set up your own server, see [here](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server). Even the server is open-source, so even if you're dealing with sensitive data, you should be good to go!

1. Install the `clearml` python package:

    ```bash
    pip install clearml
    ```

1. Connect the ClearML SDK to the server by [creating credentials](https://app.clear.ml/settings/workspace-configuration) (go right top to Settings -> Workspace -> Create new credentials), then execute the command below and follow the instructions:

    ```bash
    clearml-init
    ```

You are done, if you now run any training command like
```
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```
everything will be tracked in the ClearML server 🌟​

## 📋 Tutorial Notebook
You can see how to use ClearML as part of the tutorial notebook as well.

YOLOv5 notebook example: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

---
---
<br />

## 📈 Tracking experiments
To enable ClearML experiment tracking, simply install the package following the instruction above and you're good to go.

But there is more to the story! Take a look at the ClearML remote execution options [here](https://clear.ml/docs/latest/docs/getting_started/mlops/mlops_first_steps). Essentially, the task (an experiment in ClearML) is now reproducible!

So we can actually clone a task by right clicking it and it will be set to draft. Now you have the freedom to change any of the captured parameters and hyperparameters as you see fit. Finally, by right clicking the task again, you can enqueue the task in any of the queues on your system and a remote agent (worker) will replicate your original code, inject the parameters you changed and run the training again for you! How cool is that? 🌟​

PS: if you want to change the `project_name` or `task_name`, head over to our custom logger, where you can change it :) `utils/loggers/clearml/clearml_utils.py`

![Experiment Management Interface](https://github.com/thepycoder/clearml_screenshots/raw/main/scalars.png)

### ClearML Agents for remote execution
If you want to spin up some queues and agents (ClearML workers) yourself to remotely execute the training process of this repository, head over to our resources on the topic:

- [Youtube video](https://youtu.be/MX3BrXnaULs)
- [Documentation](https://clear.ml/docs/latest/docs/clearml_agent)
- [Example code](https://clear.ml/docs/latest/docs/guides/advanced/execute_remotely)

But in short: every experiment tracked by the experiment manager contains enough information to reproduce it on a different machine (installed packages, uncommitted changes etc.). So a ClearML agent does just that: it listens to a queue for incoming tasks and when it finds one, it recreates the environment and runs it while still reporting scalars, plots etc. to the experiment manager.

You can turn any machine (a cloud VM, a local GPU machine, your own laptop ... ) into a ClearML agent by simply running:
```
clearml-agent daemon --queue <queues_to_listen_to> [--docker]
```
Now you can clone a task like we explained above, or simply mark your current script by adding `task.execute_remotely()` and on execution it will be put into a queue, for the agent to start working on! 

### Autoscaling workers
ClearML comes with autoscalers too! This tool will automatically spin up new remote machines in the cloud of your choice (AWS, GCP, Azure) and turn them into ClearML agents for you whenever there are experiments detected in the queue. Once the tasks are processed, the autoscaler will automatically shut down the remote machines and you stop paying!

Check out the autoscalers [here](https://youtu.be/j4XVMAaUt3E).

## 🔗 Data versioning
Versioning your data separately from your code is generally a good idea. This repository supports supplying a dataset version ID and it will make sure to get the data if it's not there yet. Next to that, this workflow also saves the used dataset ID as part of the task parameters, so you will always know for sure which data was used in which experiment!

![ClearML Dataset Interface](https://github.com/thepycoder/clearml_screenshots/raw/main/dataset_version.png)

### Prepare Dataset
This repository supports a number of different datasets by using yaml files containing their information. By default datasets are downloaded to the `../datasets` folder in relation to the repository root folder. So if you downloaded the `coco128` dataset using the link in the yaml or with the scripts provided by yolov5, you get this folder structure:

```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ LICENSE
        |_ README.txt
```
But this can be any dataset you wish.

Next, ⚠️**copy the corresponding yaml file to the root of the dataset folder**⚠️. This yaml files contains the information ClearML will need to properly use the dataset. You can make this yourself too, of course, just follow the structure of the example yamls.

Basically we need the following keys: `path`, `train`, `test`, `val`, `nc`, `names`.

```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ coco128.yaml  # <---- HERE!
        |_ LICENSE
        |_ README.txt
```

### Upload dataset
To get this dataset into ClearML as a versionned dataset, go to the dataset root and run the following command:
```bash
cd datasets/coco128
clearml-data sync --project YOLOv5 --name coco128 --folder .
```

The command `clearml-data sync` is actually a shorthand command. You could also run these commands one after the other:
```bash
# Optionally add --parent <parent_dataset_id> if you want to base
# this version on another dataset version, so no duplicate files are uploaded!
clearml-data create --name coco128 --project YOLOv5
clearml-data add --files .
clearml-data close
```

### Run training using a ClearML dataset
Now that you have a ClearML dataset, you can very simply use it to train custom YOLOv5 🚀 models!

```bash
python train.py --img 640 --batch 16 --epochs 3 --data clearml:<your_dataset_id> --weights yolov5s.pt --cache
```

## ✌️ Resume execution
When training gets unexpectedly interrupted, you can easily resume a training run when you tracked it with ClearML. All the information is saved in there after all!

### Training
When training, make sure to use the `--save-period n` parameter to save a checkpoint of the model ever n iterations. If you do this, ClearML will upload the checkpoint in the background while the process continues training. So bear in mind that the latest checkpoint might not be completely uploaded yet, if the process crashes shortly after the last checkpoint was saved.

```
python train.py --img 640 --batch 16 --epochs 15 --data coco128.yaml --weights yolov5s.pt --cache --save-period 5
```

### Resuming
If the above command crashed before it completed, you can start from the last checkpoint by running this command:
```
python train.py --resume clearml:<clearml_aborted_task_id>
```

This will get the latest saved model, reset all parameters and restart training from where it left off!

## 👀 Hyperparameter optimization
To run hyperparameter optimization locally or on the cloud, we've incluided a pre-made script for you. Just make sure a training task has been run at least once, so it is in the ClearML experiment manager.

You'll need to fill in the ID of this `template task` in the script found at `utils/loggers/clearml/hpo.py` and then just run it :) You can change `task.execute_locally()` to `task.execute()` to put it in a ClearML queue and have a remote agent work on it instead.

```
python utils/loggers/clearml/hpo.py
```