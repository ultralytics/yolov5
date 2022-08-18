import os

import comet_ml
import torch
import torchvision.transforms as T
import yaml
from torchvision.utils import draw_bounding_boxes, save_image
from utils.general import scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import ConfusionMatrix, box_iou

COMET_MODE = os.getenv("COMET_MODE", "online")
COMET_SAVE_MODEL = os.getenv("COMET_SAVE_MODEL", "false").lower() == "true"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")

COMET_LOG_CONFUSION_MATRIX = (
    os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true"
)
COMET_MAX_IMAGES = os.getenv("COMET_MAX_VAL_IMAGES", 100)

COMET_OVERWRITE_CHECKPOINTS = (
    os.getenv("COMET_OVERWRITE_CHECKPOINTS", "true").lower() == "true"
)
COMET_LOG_BATCH_METRICS = (
    os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
)
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
RANK = int(os.getenv("RANK", -1))

CONF_THRES = os.getenv("CONF_THRES", 0.001)
IOU_THRES = os.getenv("IOU_THRES", 0.6)

to_pil = T.ToPILImage()


class CometLogger:
    """Log metrics, parameters, source code, models and much more
    with Comet
    """

    def __init__(
        self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs
    ) -> None:
        self.job_type = job_type
        self.opt = opt
        self.hyp = hyp

        # Comet Flags
        self.comet_mode = self.opt.comet_mode if self.opt.comet_mode else COMET_MODE
        self.save_model = (
            opt.comet_save_model if opt.comet_save_model else COMET_SAVE_MODEL
        )
        self.model_name = (
            opt.comet_model_name if opt.comet_model_name else COMET_MODEL_NAME
        )

        self.overwrite_checkpoints = (
            opt.comet_overwrite_checkpoints
            if opt.comet_overwrite_checkpoints
            else COMET_OVERWRITE_CHECKPOINTS
        )
        self.log_batch_metrics = (
            opt.comet_log_batch_metrics
            if opt.comet_log_batch_metrics
            else COMET_LOG_BATCH_METRICS
        )
        self.comet_log_batch_interval = (
            opt.comet_log_batch_interval
            if opt.comet_log_batch_interval
            else COMET_BATCH_LOGGING_INTERVAL
        )

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
        }
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)

        with open(self.opt.data, errors="ignore") as f:
            self.data_dict = yaml.safe_load(f)
        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]

        self.logged_images_count = 0
        self.max_images = (
            self.opt.comet_max_images if self.opt.comet_max_images else COMET_MAX_IMAGES
        )

        if self.experiment is not None:
            if run_id is None:
                self.log_parameters(vars(opt))
                self.log_asset(opt.hyp, metadata={"type": "hyp-config-file"})
                self.log_asset(
                    f"{opt.save_dir}/opt.yaml", metadata={"type": "opt-config-file"}
                )
                self.experiment.log_other("Created from", "YOLOv5")
                self.experiment.log_other(
                    "Run ID",
                    f"{self.experiment.workspace}/{self.experiment.project_name}/{self.experiment.id}",
                )

        self.comet_log_confusion_matrix = (
            self.opt.comet_log_confusion_matrix
            if self.opt.comet_log_confusion_matrix
            else COMET_LOG_CONFUSION_MATRIX
        )
        if self.comet_log_confusion_matrix:
            self.conf_thres = CONF_THRES
            self.iou_thres = IOU_THRES

        self.comet_log_predictions = True
        if self.comet_log_predictions:
            self.metadata_dict = {}

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                **self.default_experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.Experiment(
                **self.default_experiment_kwargs,
            )

        return

    def log_metrics(self, log_dict, **kwargs):
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        self.experiment.log_asset(asset_path, **kwargs)

    def log_image(self, img, **kwargs):
        self.experiment.log_image(img, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        if not self.save_model:
            return

        model_metadata = {
            "fitness_score": fitness_score,
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }

        if opt.comet_checkpoint_filename is "all":
            model_path = str(path)
        else:
            model_path = str(path) + f"/{opt.comet_checkpoint_filename}"

        self.experiment.log_model(
            self.model_name,
            model_path,
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
            metadata.append(
                {
                    "label": f"{self.class_names[int(cls)]}-gt",
                    "score": 100,
                    "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                }
            )
        for *xyxy, conf, cls in filtered_detections.tolist():
            metadata.append(
                {
                    "label": f"{self.class_names[int(cls)]}",
                    "score": conf * 100,
                    "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                }
            )

        self.metadata_dict[image_name] = metadata
        self.logged_images_count += 1

        return

    def preprocess_prediction(self, image, labels, shape, pred):
        nl, npr = labels.shape[0], pred.shape[0]

        # Predictions
        if self.opt.single_cls:
            pred[:, 5] = 0

        predn = pred.clone()
        scale_coords(image.shape[1:], predn[:, :4], shape[0], shape[1])

        labelsn = None
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(
                image.shape[1:], tbox, shape[0], shape[1]
            )  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            scale_coords(
                image.shape[1:], predn[:, :4], shape[0], shape[1]
            )  # native-space pred

        return predn, labelsn

    def on_pretrain_routine_start(self):
        return

    def on_pretrain_routine_end(self, paths):
        for path in paths:
            self.log_asset(str(path))

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

    def on_train_end(self, files, save_dir, epoch):
        if self.comet_log_predictions:
            curr_epoch = self.experiment.curr_epoch
            self.experiment.log_asset_data(
                self.metadata_dict, "image-metadata.json", epoch=curr_epoch
            )

        for f in files:
            self.log_asset(f, metadata={"epoch": epoch})
        self.log_asset(f"{save_dir}/results.csv", metadata={"epoch": epoch})
        self.finish_run()

    def on_val_start(self):
        self.confmat = ConfusionMatrix(
            nc=self.num_classes, conf=self.conf_thres, iou_thres=self.iou_thres
        )

        return

    def on_val_batch_start(self):
        return

    def on_val_batch_end(self, images, targets, paths, shapes, outputs):
        for si, pred in enumerate(outputs):
            if len(pred) == 0:
                continue

            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            shape = shapes[si]
            path = paths[si]

            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)

            if self.comet_log_predictions and (
                self.experiment.curr_step % self.comet_log_batch_interval == 0
            ):
                if labelsn is not None:
                    self.log_predictions(image, labelsn, path, shape, predn)

            if self.comet_log_confusion_matrix:
                if labelsn is not None:
                    self.confmat.process_batch(predn, labelsn)

        return

    def on_fit_epoch_end(self, result, epoch):
        self.log_metrics(result, epoch=epoch)

        if self.comet_log_confusion_matrix:
            self.experiment.log_confusion_matrix(
                matrix=self.confmat.matrix[: self.num_classes, : self.num_classes],
                max_categories=self.num_classes,
                labels=self.class_names,
                epoch=epoch,
            )

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if (
            (epoch + 1) % self.opt.save_period == 0 and not final_epoch
        ) and self.opt.save_period != -1:
            self.comet_logger.log_model(
                last.parent, self.opt, epoch, fi, best_model=best_fitness == fi
            )

    def finish_run(self):
        if self.experiment is not None:
            self.experiment.end()
