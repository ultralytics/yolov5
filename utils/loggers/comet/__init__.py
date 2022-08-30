import glob
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import comet_ml

    # Project Configuration
    config = comet_ml.config.get_config()
    COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")
except (ModuleNotFoundError, ImportError):
    comet_ml = None
    COMET_PROJECT_NAME = None

import torch
import torchvision.transforms as T
import yaml

from utils.dataloaders import img2label_paths
from utils.general import check_dataset, scale_coords, xywh2xyxy
from utils.metrics import ConfusionMatrix, box_iou

COMET_MODE = os.getenv("COMET_MODE", "online")

# Model Saving Settings
COMET_SAVE_MODEL = os.getenv("COMET_SAVE_MODEL", "false").lower() == "true"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_OVERWRITE_CHECKPOINTS = (os.getenv("COMET_OVERWRITE_CHECKPOINTS", "false").lower() == "true")

# Dataset Artifact Settings
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"

# Evaluation Settings
COMET_LOG_CONFUSION_MATRIX = (os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true")
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "false").lower() == "true"
COMET_MAX_IMAGE_UPLOADS = os.getenv("COMET_MAX_IMAGE_UPLOADS", 100)

# Confusion Matrix Settings
CONF_THRES = os.getenv("CONF_THRES", 0.001)
IOU_THRES = os.getenv("IOU_THRES", 0.6)

# Batch Logging Settings
COMET_LOG_BATCH_METRICS = (os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true")
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
COMET_PREDICTION_LOGGING_INTERVAL = os.getenv("COMET_PREDICTION_LOGGING_INTERVAL", 1)

RANK = int(os.getenv("RANK", -1))

to_pil = T.ToPILImage()


class CometLogger:
    """Log metrics, parameters, source code, models and much more
    with Comet
    """

    def __init__(self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs) -> None:
        self.job_type = job_type
        self.opt = opt
        self.hyp = hyp

        # Comet Flags
        self.comet_mode = self.opt.comet_mode if self.opt.comet_mode else COMET_MODE

        self.save_model = (opt.comet_save_model if opt.comet_save_model else COMET_SAVE_MODEL)
        self.model_name = (opt.comet_model_name if opt.comet_model_name else COMET_MODEL_NAME)
        self.overwrite_checkpoints = (opt.comet_overwrite_checkpoints
                                      if opt.comet_overwrite_checkpoints else COMET_OVERWRITE_CHECKPOINTS)

        # Batch Logging Settings
        self.log_batch_metrics = (opt.comet_log_batch_metrics
                                  if opt.comet_log_batch_metrics else COMET_LOG_BATCH_METRICS)
        self.comet_log_batch_interval = (opt.comet_log_batch_interval
                                         if opt.comet_log_batch_interval else COMET_BATCH_LOGGING_INTERVAL)

        # Dataset Artifact Settings
        self.upload_dataset = (self.opt.comet_upload_dataset if self.opt.comet_upload_dataset else COMET_UPLOAD_DATASET)
        self.resume = self.opt.resume

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMET_PROJECT_NAME,}
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)

        if self.opt.comet_artifact:
            self.data_dict = self.download_dataset_artifact(self.opt.comet_artifact)

        else:
            self.data_dict = check_dataset(self.opt.data)

        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]

        self.logged_images_count = 0
        self.max_images = (self.opt.comet_max_image_uploads
                           if self.opt.comet_max_image_uploads else COMET_MAX_IMAGE_UPLOADS)
        if self.experiment is not None:
            if run_id is None:
                self.experiment.log_other("Created from", "YOLOv5")

                if not isinstance(self.experiment, comet_ml.OfflineExperiment):
                    workspace, project_name, experiment_id = self.experiment.url.split("/")[-3:]
                    self.experiment.log_other(
                        "Run Path",
                        f"{workspace}/{project_name}/{experiment_id}",
                    )
                self.log_parameters(vars(opt))
                self.log_parameters(self.opt.hyp)
                self.log_asset_data(
                    self.opt.hyp,
                    name="hyperparameters.json",
                    metadata={"type": "hyp-config-file"},
                )
                self.log_asset(
                    f"{self.opt.save_dir}/opt.yaml",
                    metadata={"type": "opt-config-file"},
                )
                if not self.opt.comet_artifact:
                    self.log_asset(self.opt.data, metadata={"type": "data-config-file"})

        self.comet_log_confusion_matrix = (self.opt.comet_log_confusion_matrix
                                           if self.opt.comet_log_confusion_matrix else COMET_LOG_CONFUSION_MATRIX)
        if self.comet_log_confusion_matrix:
            if hasattr(self.opt, "conf_thres"):
                self.conf_thres = self.opt.conf_thres
            else:
                self.conf_thres = CONF_THRES
            if hasattr(self.opt, "iou_thres"):
                self.iou_thres = self.opt.iou_thres
            else:
                self.iou_thres = IOU_THRES

        self.comet_log_predictions = (self.opt.comet_log_predictions
                                      if self.opt.comet_log_predictions else COMET_LOG_PREDICTIONS)
        self.comet_log_prediction_interval = (opt.comet_log_prediction_interval if opt.comet_log_prediction_interval
                                              else COMET_PREDICTION_LOGGING_INTERVAL)
        if self.comet_log_predictions:
            self.metadata_dict = {}

        # Check if running the Experiment with the Comet Optimizer
        if hasattr(self.opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", self.opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", self.opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters", json.dumps(self.hyp))

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(**self.default_experiment_kwargs,)

        else:
            try:
                if experiment_id is not None:
                    return comet_ml.ExistingExperiment(
                        previous_experiment=experiment_id,
                        **self.default_experiment_kwargs,
                    )

                return comet_ml.Experiment(**self.default_experiment_kwargs,)

            except ValueError as e:
                logger.warning("COMET WARNING: "
                               "Comet credentials have not been set. "
                               "Comet will default to offline logging. "
                               "Please set your credentials to enable online logging.")
                logger.exceptiom(e)
                return self._get_experiment("offline", experiment_id)

        return

    def log_metrics(self, log_dict, **kwargs):
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        self.experiment.log_asset(asset_path, **kwargs)

    def log_asset_data(self, asset, **kwargs):
        self.experiment.log_asset_data(asset, **kwargs)

    def log_image(self, img, **kwargs):
        self.experiment.log_image(img, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        if not self.save_model:
            return

        model_metadata = {
            "fitness_score": fitness_score[-1],
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,}

        if opt.comet_checkpoint_filename == "all":
            model_path = str(path)
            model_files = glob.glob(f"{path}/*.pt")

        else:
            model_files = [str(path) + f"/{opt.comet_checkpoint_filename}"]

        for model_path in model_files:
            if not self.overwrite_checkpoints:
                name = f"{Path(model_path).stem}_epoch_{epoch}.pt"
            else:
                name = Path(model_path).name

            self.experiment.log_model(
                self.model_name,
                file_or_folder=model_path,
                file_name=name,
                metadata=model_metadata,
                overwrite=self.overwrite_checkpoints,
            )

    def log_predictions(self, image, labelsn, path, shape, predn):
        if self.logged_images_count >= self.max_images:
            return

        detections = predn[predn[:, 4] > self.conf_thres]
        iou = box_iou(labelsn[:, 1:], detections[:, :4])
        mask, _ = torch.where(iou > self.iou_thres)
        if len(mask) == 0:
            return

        filtered_detections = detections[mask]
        filtered_labels = labelsn[mask]

        processed_image = (image * 255).to(torch.uint8)

        image_id = path.split("/")[-1].split(".")[0]
        image_name = f"{image_id}_curr_epoch_{self.experiment.curr_epoch}"
        self.log_image(to_pil(processed_image), name=image_name)

        metadata = []
        for cls, *xyxy in filtered_labels.tolist():
            metadata.append({
                "label": f"{self.class_names[int(cls)]}-gt",
                "score": 100,
                "box": {
                    "x": xyxy[0],
                    "y": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3]},})
        for *xyxy, conf, cls in filtered_detections.tolist():
            metadata.append({
                "label": f"{self.class_names[int(cls)]}",
                "score": conf * 100,
                "box": {
                    "x": xyxy[0],
                    "y": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3]},})

        self.metadata_dict[image_name] = metadata
        self.logged_images_count += 1

        return

    def preprocess_prediction(self, image, labels, shape, pred):
        nl, _ = labels.shape[0], pred.shape[0]

        # Predictions
        if self.opt.single_cls:
            pred[:, 5] = 0

        predn = pred.clone()
        scale_coords(image.shape[1:], predn[:, :4], shape[0], shape[1])

        labelsn = None
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(image.shape[1:], tbox, shape[0], shape[1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            scale_coords(image.shape[1:], predn[:, :4], shape[0], shape[1])  # native-space pred

        return predn, labelsn

    def add_assets_to_artifact(self, artifact, path, asset_path, split):
        img_paths = sorted(glob.glob(f"{asset_path}/*"))
        label_paths = img2label_paths(img_paths)

        for image_file, label_file in zip(img_paths, label_paths):
            image_logical_path, label_logical_path = map(lambda x: x.replace(f"{path}/", ""), [image_file, label_file])
            artifact.add(image_file, logical_path=image_logical_path, metadata={"split": split})
            artifact.add(label_file, logical_path=label_logical_path, metadata={"split": split})
        return artifact

    def upload_dataset_artifact(self):
        dataset_name = self.data_dict.get("dataset_name", "yolov5-dataset")
        path = self.data_dict["path"]

        metadata = self.data_dict.copy()
        for key in ["train", "val", "test"]:
            split_path = metadata.get(key)
            if split_path is not None:
                metadata[key] = split_path.replace(path, "")

        artifact = comet_ml.Artifact(name=dataset_name, artifact_type="dataset", metadata=metadata)
        for key in metadata.keys():
            if key in ["train", "val", "test"]:
                if isinstance(self.upload_dataset, str) and (key != self.upload_dataset):
                    continue

                asset_path = self.data_dict.get(key)
                if asset_path is not None:
                    artifact = self.add_assets_to_artifact(artifact, path, asset_path, key)

        self.experiment.log_artifact(artifact)

        return

    def download_dataset_artifact(self, artifact_path):
        logged_artifact = self.experiment.get_artifact(artifact_path)
        logged_artifact.download(self.opt.save_dir)

        metadata = logged_artifact.metadata
        data_dict = metadata.copy()
        data_dict["path"] = self.opt.save_dir
        data_dict["names"] = {int(k): v for k, v in metadata.get("names").items()}

        data_dict = self.update_data_paths(data_dict)
        return data_dict

    def update_data_paths(self, data_dict):
        path = data_dict.get("path", "")

        for split in ["train", "val", "test"]:
            if data_dict.get(split):
                split_path = data_dict.get(split)
                data_dict[split] = (f"{path}/{split_path}" if isinstance(split, str) else [
                    f"{path}/{x}" for x in split_path])

        return data_dict

    def on_pretrain_routine_end(self, paths):
        if self.opt.resume:
            return

        for path in paths:
            self.log_asset(str(path))

        if self.upload_dataset:
            if not self.resume:
                self.upload_dataset_artifact()

        return

    def on_train_start(self):
        self.log_parameters(self.hyp)

    def on_train_epoch_start(self):
        return

    def on_train_epoch_end(self, epoch):
        self.experiment.curr_epoch = epoch

        return

    def on_train_batch_start(self):
        return

    def on_train_batch_end(self, log_dict, step):
        self.experiment.curr_step = step
        if self.log_batch_metrics and (step % self.comet_log_batch_interval == 0):
            self.log_metrics(log_dict, step=step)

        return

    def on_train_end(self, files, save_dir, last, best, epoch, results):
        if self.comet_log_predictions:
            curr_epoch = self.experiment.curr_epoch
            self.experiment.log_asset_data(self.metadata_dict, "image-metadata.json", epoch=curr_epoch)

        for f in files:
            self.log_asset(f, metadata={"epoch": epoch})
        self.log_asset(f"{save_dir}/results.csv", metadata={"epoch": epoch})

        if not self.opt.evolve:
            model_path = str(best if best.exists() else last)
            if not self.overwrite_checkpoints:
                name = name = f"{Path(model_path).stem}_epoch_{epoch}.pt"
            else:
                name = Path(model_path).name
            if self.save_model:
                self.experiment.log_model(
                    self.model_name,
                    file_or_folder=model_path,
                    file_name=name,
                    overwrite=self.overwrite_checkpoints,
                )

        # Check if running Experiment with Comet Optimizer
        if hasattr(self.opt, 'comet_optimizer_id'):
            metric = results.get(self.opt.comet_optimizer_metric)
            self.experiment.log_other('optimizer_metric_value', metric)

        self.finish_run()

    def on_val_start(self):
        self.confmat = ConfusionMatrix(nc=self.num_classes, conf=self.conf_thres, iou_thres=self.iou_thres)

        return

    def on_val_batch_start(self):
        return

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
        for si, pred in enumerate(outputs):
            if len(pred) == 0:
                continue

            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            shape = shapes[si]
            path = paths[si]

            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)
            if labelsn is not None:
                if self.comet_log_predictions and ((batch_i + 1) % self.comet_log_prediction_interval == 0):
                    self.log_predictions(image, labelsn, path, shape, predn)

                if self.comet_log_confusion_matrix:
                    self.confmat.process_batch(predn, labelsn)

        return

    def on_fit_epoch_end(self, result, epoch):
        self.log_metrics(result, epoch=epoch)

        if self.comet_log_confusion_matrix:
            class_names = list(self.class_names.values())
            class_names.append("background-FN")

            num_classes = len(class_names)

            self.experiment.log_confusion_matrix(
                matrix=self.confmat.matrix,
                max_categories=num_classes,
                labels=class_names,
                epoch=epoch,
                file_name=f"confusion-matrix-epoch-{epoch}.json",
            )

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
            self.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def finish_run(self):
        if self.experiment is not None:
            self.experiment.end()
