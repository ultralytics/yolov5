import glob
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

try:
    import comet_ml

    # Project Configuration
    config = comet_ml.config.get_config()
    COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")
except (ModuleNotFoundError, ImportError):
    comet_ml = None
    COMET_PROJECT_NAME = None

import PIL
import torch
import torchvision.transforms as T
import yaml

from utils.dataloaders import img2label_paths
from utils.general import check_dataset, scale_boxes, xywh2xyxy
from utils.metrics import box_iou

COMET_PREFIX = "comet://"

COMET_MODE = os.getenv("COMET_MODE", "online")

# Model Saving Settings
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")

# Dataset Artifact Settings
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"

# Evaluation Settings
COMET_LOG_CONFUSION_MATRIX = os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true"
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "true").lower() == "true"
COMET_MAX_IMAGE_UPLOADS = int(os.getenv("COMET_MAX_IMAGE_UPLOADS", 100))

# Confusion Matrix Settings
CONF_THRES = float(os.getenv("CONF_THRES", 0.001))
IOU_THRES = float(os.getenv("IOU_THRES", 0.6))

# Batch Logging Settings
COMET_LOG_BATCH_METRICS = os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
COMET_PREDICTION_LOGGING_INTERVAL = os.getenv("COMET_PREDICTION_LOGGING_INTERVAL", 1)
COMET_LOG_PER_CLASS_METRICS = os.getenv("COMET_LOG_PER_CLASS_METRICS", "false").lower() == "true"

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
        self.comet_mode = COMET_MODE

        self.save_model = opt.save_period > -1
        self.model_name = COMET_MODEL_NAME

        # Batch Logging Settings
        self.log_batch_metrics = COMET_LOG_BATCH_METRICS
        self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL

        # Dataset Artifact Settings
        self.upload_dataset = self.opt.upload_dataset if self.opt.upload_dataset else COMET_UPLOAD_DATASET
        self.resume = self.opt.resume

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMET_PROJECT_NAME,}
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)

        self.data_dict = self.check_dataset(self.opt.data)
        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]

        self.logged_images_count = 0
        self.max_images = COMET_MAX_IMAGE_UPLOADS

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

        self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX

        if hasattr(self.opt, "conf_thres"):
            self.conf_thres = self.opt.conf_thres
        else:
            self.conf_thres = CONF_THRES
        if hasattr(self.opt, "iou_thres"):
            self.iou_thres = self.opt.iou_thres
        else:
            self.iou_thres = IOU_THRES

        self.log_parameters({"val_iou_threshold": self.iou_thres, "val_conf_threshold": self.conf_thres})

        self.comet_log_predictions = COMET_LOG_PREDICTIONS
        if self.opt.bbox_interval == -1:
            self.comet_log_prediction_interval = 1 if self.opt.epochs < 10 else self.opt.epochs // 10
        else:
            self.comet_log_prediction_interval = self.opt.bbox_interval

        if self.comet_log_predictions:
            self.metadata_dict = {}
            self.logged_image_names = []

        self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS

        self.experiment.log_others({
            "comet_mode": COMET_MODE,
            "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
            "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
            "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
            "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
            "comet_model_name": COMET_MODEL_NAME,})

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

                return comet_ml.Experiment(**self.default_experiment_kwargs)

            except ValueError:
                logger.warning("COMET WARNING: "
                               "Comet credentials have not been set. "
                               "Comet will default to offline logging. "
                               "Please set your credentials to enable online logging.")
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

        model_files = glob.glob(f"{path}/*.pt")
        for model_path in model_files:
            name = Path(model_path).name

            self.experiment.log_model(
                self.model_name,
                file_or_folder=model_path,
                file_name=name,
                metadata=model_metadata,
                overwrite=True,
            )

    def check_dataset(self, data_file):
        with open(data_file) as f:
            data_config = yaml.safe_load(f)

        if data_config['path'].startswith(COMET_PREFIX):
            path = data_config['path'].replace(COMET_PREFIX, "")
            data_dict = self.download_dataset_artifact(path)

            return data_dict

        self.log_asset(self.opt.data, metadata={"type": "data-config-file"})

        return check_dataset(data_file)

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

        image_id = path.split("/")[-1].split(".")[0]
        image_name = f"{image_id}_curr_epoch_{self.experiment.curr_epoch}"
        if image_name not in self.logged_image_names:
            native_scale_image = PIL.Image.open(path)
            self.log_image(native_scale_image, name=image_name)
            self.logged_image_names.append(image_name)

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
        scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])

        labelsn = None
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(image.shape[1:], tbox, shape[0], shape[1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])  # native-space pred

        return predn, labelsn

    def add_assets_to_artifact(self, artifact, path, asset_path, split):
        img_paths = sorted(glob.glob(f"{asset_path}/*"))
        label_paths = img2label_paths(img_paths)

        for image_file, label_file in zip(img_paths, label_paths):
            image_logical_path, label_logical_path = map(lambda x: os.path.relpath(x, path), [image_file, label_file])

            try:
                artifact.add(image_file, logical_path=image_logical_path, metadata={"split": split})
                artifact.add(label_file, logical_path=label_logical_path, metadata={"split": split})
            except ValueError as e:
                logger.error('COMET ERROR: Error adding file to Artifact. Skipping file.')
                logger.error(f"COMET ERROR: {e}")
                continue

        return artifact

    def upload_dataset_artifact(self):
        dataset_name = self.data_dict.get("dataset_name", "yolov5-dataset")
        path = str((ROOT / Path(self.data_dict["path"])).resolve())

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
        artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
        logged_artifact.download(artifact_save_dir)

        metadata = logged_artifact.metadata
        data_dict = metadata.copy()
        data_dict["path"] = artifact_save_dir
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
            name = Path(model_path).name
            if self.save_model:
                self.experiment.log_model(
                    self.model_name,
                    file_or_folder=model_path,
                    file_name=name,
                    overwrite=True,
                )

        # Check if running Experiment with Comet Optimizer
        if hasattr(self.opt, 'comet_optimizer_id'):
            metric = results.get(self.opt.comet_optimizer_metric)
            self.experiment.log_other('optimizer_metric_value', metric)

        self.finish_run()

    def on_val_start(self):
        return

    def on_val_batch_start(self):
        return

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
        if not (self.comet_log_predictions and ((batch_i + 1) % self.comet_log_prediction_interval == 0)):
            return

        for si, pred in enumerate(outputs):
            if len(pred) == 0:
                continue

            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            shape = shapes[si]
            path = paths[si]
            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)
            if labelsn is not None:
                self.log_predictions(image, labelsn, path, shape, predn)

        return

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        if self.comet_log_per_class_metrics:
            if self.num_classes > 1:
                for i, c in enumerate(ap_class):
                    class_name = self.class_names[c]
                    self.experiment.log_metrics(
                        {
                            'mAP@.5': ap50[i],
                            'mAP@.5:.95': ap[i],
                            'precision': p[i],
                            'recall': r[i],
                            'f1': f1[i],
                            'true_positives': tp[i],
                            'false_positives': fp[i],
                            'support': nt[c]},
                        prefix=class_name)

        if self.comet_log_confusion_matrix:
            epoch = self.experiment.curr_epoch
            class_names = list(self.class_names.values())
            class_names.append("background")
            num_classes = len(class_names)

            self.experiment.log_confusion_matrix(
                matrix=confusion_matrix.matrix,
                max_categories=num_classes,
                labels=class_names,
                epoch=epoch,
                column_label='Actual Category',
                row_label='Predicted Category',
                file_name=f"confusion-matrix-epoch-{epoch}.json",
            )

    def on_fit_epoch_end(self, result, epoch):
        self.log_metrics(result, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
            self.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_params_update(self, params):
        self.log_parameters(params)

    def finish_run(self):
        self.experiment.end()
