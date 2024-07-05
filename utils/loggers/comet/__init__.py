# Ultralytics YOLOv5 🚀, AGPL-3.0 license

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
except ImportError:
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
    """Log metrics, parameters, source code, models and much more with Comet."""

    def __init__(self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs) -> None:
        """
        Initializes the CometLogger with experiment parameters and configurations.
        
        Args:
            opt (Namespace): The options containing configurations such as save periods, resume, and dataset upload flags.
            hyp (dict): The hyperparameters dictionary.
            run_id (str | None): Unique identifier for resuming an existing experiment. Defaults to None.
            job_type (str): The type of job, e.g., "Training". Defaults to "Training".
            **experiment_kwargs: Additional keyword arguments to configure comet_ml.Experiment objects.
        
        Returns:
            None
        
        Notes:
            The CometLogger integrates with Comet ML to log metadata, metrics, hyperparameters, and artifacts for ML 
            experiments. To use this logger, ensure Comet ML is properly installed and configured. 
        
        Example:
            ```python
            from namespace import Namespace
            import yaml
        
            with open('hyp.yaml') as f:
                hyp = yaml.load(f, Loader=yaml.FullLoader)
        
            opt = Namespace(save_period=1, resume=False, upload_dataset=True, name='run1', data='data.yaml')
            
            logger = CometLogger(opt, hyp)
            ```
        """
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
        self.upload_dataset = self.opt.upload_dataset or COMET_UPLOAD_DATASET
        self.resume = self.opt.resume

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMET_PROJECT_NAME,
        }
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)
        self.experiment.set_name(self.opt.name)

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

        self.experiment.log_others(
            {
                "comet_mode": COMET_MODE,
                "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
                "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
                "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
                "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
                "comet_model_name": COMET_MODEL_NAME,
            }
        )

        # Check if running the Experiment with the Comet Optimizer
        if hasattr(self.opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", self.opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", self.opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters", json.dumps(self.hyp))

    def _get_experiment(self, mode, experiment_id=None):
        """
        _get_experiment(mode: str, experiment_id: str | None = None) -> comet_ml.Experiment | comet_ml.OfflineExperiment
        Returns a new or existing Comet.ml experiment based on the mode and optional experiment ID.
        
        Args:
            mode (str): Mode of the experiment, either "online" or "offline".
            experiment_id (str | None): Identifier for an existing experiment. If None, a new experiment is created.
        
        Returns:
            comet_ml.Experiment | comet_ml.ExistingExperiment: An instance of a Comet.ml Experiment or OfflineExperiment.
        """
        if mode == "offline":
            return (
                comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )
                if experiment_id is not None
                else comet_ml.OfflineExperiment(
                    **self.default_experiment_kwargs,
                )
            )
        try:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.Experiment(**self.default_experiment_kwargs)

        except ValueError:
            logger.warning(
                "COMET WARNING: "
                "Comet credentials have not been set. "
                "Comet will default to offline logging. "
                "Please set your credentials to enable online logging."
            )
            return self._get_experiment("offline", experiment_id)

        return

    def log_metrics(self, log_dict, **kwargs):
        """
        Logs metrics to the current Comet experiment.
        
        Args:
            log_dict (dict): Dictionary containing metric names as keys and their corresponding values.
            **kwargs: Additional keyword arguments to be passed to Comet's log_metrics method.
        
        Returns:
            None
        
        Notes:
            This method utilizes the Comet.ml API to log metrics during training, validation, or testing phases.
            Ensure that Comet API is properly configured and authenticated to enable logging.
        """
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        """
        Logs parameters to the current Comet.ml experiment, accepting a dictionary of parameter names and values.
        
        Args:
            log_dict (dict): Dictionary containing the parameter names and values to be logged.
            **kwargs: Additional keyword arguments to pass to `comet_ml.Experiment.log_parameters()` method.
        
        Returns:
            None
        
        Notes:
            - Parameters are vital hyperparameters and configurations that influence the training and evaluation processes.
            - Ensure that `log_dict` contains meaningful and correctly-named parameters for effective tracking and reproducibility.
        
        Examples:
        ```python
        logger = CometLogger(opt, hyp)
        params = {'learning_rate': 0.001, 'batch_size': 64}
        logger.log_parameters(params)
        ```
        """
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        """
        Logs a file or directory as an asset to the current Comet.ml experiment.
        
        Args:
            asset_path (str | Path): Path to the file or directory to be logged as an asset.
            kwargs (dict): Additional arguments to pass to the comet_ml.Experiment.log_asset method, such as metadata or
                file type.
        
        Returns:
            None
            
        Notes:
            For more information on logging assets with Comet.ml, see
            `Comet.ml Documentation <https://www.comet.ml/docs/python-sdk/Experiment/#log_asset>`_.
        
        Example:
            ```python
            logger = CometLogger(opt, hyp)
            logger.log_asset("path/to/asset/file_or_directory")
            ```
        """
        self.experiment.log_asset(asset_path, **kwargs)

    def log_asset_data(self, asset, **kwargs):
        """
        Logs in-memory data as an asset to the current Comet.ml experiment, with optional metadata and asset name.
        
        Args:
            asset (Union[str, dict]): The actual data you want to log as an asset. Typically it can be a JSON string or a dictionary.
            name (str, optional): Optional name to identify the asset within the experiment. Defaults to None.
            metadata (dict, optional): Optional metadata dictionary associated with the asset. Defaults to None.
            **kwargs: Additional keyword arguments for customization.
                
        Returns:
            None
        
        Example:
            ```python
            logger.log_asset_data(
                json.dumps({"model": "YOLOv5", "accuracy": 0.91}),
                name="experiment_results.json",
                metadata={"type": "evaluation"}
            )
            ```
        """
        self.experiment.log_asset_data(asset, **kwargs)

    def log_image(self, img, **kwargs):
        """
        Log an image to the current Comet experiment with optional metadata.
        
        Args:
            img (PIL.Image.Image | str): The image to log. This can be a PIL Image object or a file path to the image.
            kwargs (dict): Additional keyword arguments to pass to `comet_ml.Experiment.log_image`. For example, 
                           `name` (str) to specify the image name, `step` (int) to specify the step at which the image 
                           was logged, and `metadata` (dict) for additional metadata related to the image.
        
        Returns:
            None
        
        Examples:
            ```python
            from PIL import Image
            from ultralytics import CometLogger
        
            # Initialize CometLogger
            logger = CometLogger(opt, hyp)
        
            # Log an image from a file path
            logger.log_image('path/to/image.jpg', name='sample_image', step=1)
        
            # Log a PIL image
            img = Image.open('path/to/image.jpg')
            logger.log_image(img, name='sample_image', step=1)
            ```
        """
        self.experiment.log_image(img, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Logs the model checkpoint to the Comet experiment, including optional metadata such as fitness score, 
        epoch, and whether it is the best model.
        
        Args:
            path (str): The directory path where the model checkpoint files are saved.
            opt (argparse.Namespace): The parsed command-line arguments containing model training options.
            epoch (int): The current training epoch.
            fitness_score (list[float]): A list containing fitness scores for the current training epoch.
            best_model (bool): Flag to indicate if the current model is the best model (default is False).
        
        Returns:
            None
        """
        if not self.save_model:
            return

        model_metadata = {
            "fitness_score": fitness_score[-1],
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }

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
        """
        Validates the dataset configuration by loading the YAML file specified in `data_file`, ensuring proper loading
        and handling of Comet dataset artifacts if necessary.
        
        Args:
            data_file (str): Path to the dataset configuration file in YAML format.
        
        Returns:
            dict: Dataset configuration dictionary. Contains loaded dataset information.
        
        Notes:
            If the dataset path in the configuration file starts with 'comet://', it will be treated as a Comet dataset artifact
            and the respective function to download the dataset will be invoked.
        """
        with open(data_file) as f:
            data_config = yaml.safe_load(f)

        path = data_config.get("path")
        if path and path.startswith(COMET_PREFIX):
            path = data_config["path"].replace(COMET_PREFIX, "")
            return self.download_dataset_artifact(path)
        self.log_asset(self.opt.data, metadata={"type": "data-config-file"})

        return check_dataset(data_file)

    def log_predictions(self, image, labelsn, path, shape, predn):
        """
        Log predictions with IoU filtering, given image, labels, path, shape, and predictions.
        
        Args:
            image (Tensor): The input image tensor.
            labelsn (torch.Tensor): Ground truth labels in the dataset.
            path (str): Path to the image file.
            shape (tuple): Shape of the original image.
            predn (torch.Tensor): Model predictions as bounding boxes and confidence scores.
        
        Returns:
            None: No explicit return value; logs relevant information to the Comet experiment if conditions are met.
        
        Notes:
            - Filters predictions using the confidence and IoU thresholds defined during initialization.
            - Logs images and associated metadata to the current experiment, respecting the maximum image upload limit.
            - Ensures images are logged only once per epoch to avoid redundant uploads.
        
        Example:
            ```python
            # Initialize CometLogger
            comet_logger = CometLogger(opt, hyp)
            
            # Log predictions
            comet_logger.log_predictions(image, labelsn, path, shape, predn)
            ```
        """
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

        metadata = [
            {
                "label": f"{self.class_names[int(cls)]}-gt",
                "score": 100,
                "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            }
            for cls, *xyxy in filtered_labels.tolist()
        ]
        metadata.extend(
            {
                "label": f"{self.class_names[int(cls)]}",
                "score": conf * 100,
                "box": {"x": xyxy[0], "y": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
            }
            for *xyxy, conf, cls in filtered_detections.tolist()
        )
        self.metadata_dict[image_name] = metadata
        self.logged_images_count += 1

        return

    def preprocess_prediction(self, image, labels, shape, pred):
        """
        Processes prediction data, resizing labels and adding dataset metadata.
        
        Args:
            image (np.ndarray): Input image as a NumPy array.
            labels (torch.Tensor): Ground truth labels for the image, with shape (n_labels, 5).
            shape (tuple): Original shape of the image as (height, width).
            pred (torch.Tensor): Predictions from the model, with shape (n_predictions, 6).
        
        Returns:
            tuple: A tuple containing:
                - predn (torch.Tensor): Processed predictions with resized bounding boxes.
                - labelsn (torch.Tensor | None): Resized ground truth labels if any, otherwise None.
        
        Notes:
            The function processes predictions by scaling bounding boxes to match the original image shape.
            Additionally, it handles single-class scenarios by setting class indices in predictions to zero,
            and prepares labels in native-space coordinates.
        """
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
        """
        Adds image and label assets to a Comet artifact for a specified dataset split.
        
        Args:
            artifact (comet_ml.Artifact): The Comet artifact to which assets are added.
            path (str): Root path of the dataset.
            asset_path (str): Path to the asset directory containing images and labels.
            split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        
        Returns:
            None
        
        Notes:
            This function searches for image files and corresponding label files within `asset_path`, then adds them to 
            the `artifact` with logical paths relative to `path`. Metadata about the dataset split is also attached to each 
            added file.
        
        Example:
            ```python
            artifact = comet_ml.Artifact(name="dataset", artifact_type="dataset")
            add_assets_to_artifact(artifact, "/path/to/dataset", "/path/to/assets", "train")
            ```
        """
        img_paths = sorted(glob.glob(f"{asset_path}/*"))
        label_paths = img2label_paths(img_paths)

        for image_file, label_file in zip(img_paths, label_paths):
            image_logical_path, label_logical_path = map(lambda x: os.path.relpath(x, path), [image_file, label_file])

            try:
                artifact.add(
                    image_file,
                    logical_path=image_logical_path,
                    metadata={"split": split},
                )
                artifact.add(
                    label_file,
                    logical_path=label_logical_path,
                    metadata={"split": split},
                )
            except ValueError as e:
                logger.error("COMET ERROR: Error adding file to Artifact. Skipping file.")
                logger.error(f"COMET ERROR: {e}")
                continue

        return artifact

    def upload_dataset_artifact(self):
        """
        Uploads a YOLOv5 dataset as an artifact to the Comet.ml platform.
        
        Args:
            None
        
        Returns:
            None
        
        Notes:
            This function packages and uploads the dataset specified in the `self.data_dict` attribute of the CometLogger 
            instance to Comet.ml as a dataset artifact. The dataset name, paths, and metadata are derived from the dataset's
            configuration YAML.
        
        Examples:
            ```python
            logger = CometLogger(opt, hyp)
            logger.upload_dataset_artifact()
            ```
        """
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
        """
        Downloads a dataset artifact to a specified directory using the experiment's logged artifact.
        
        Args:
            artifact_path (str): The path to the Comet.ml artifact.
        
        Returns:
            dict: A dictionary containing the dataset configuration with the local path updated to the download location.
        
        Raises:
            ValueError: If the 'names' field in the dataset metadata is not a list or dictionary.
        
        Notes:
            Ensure that the Comet.ml experiment is properly initialized and connected before calling this function.
            The dataset configuration will be updated with the local path after downloading the artifact.
        
        Example:
            ```python
            artifact_path = "comet://path/to/artifact"
            dataset_config = comet_logger.download_dataset_artifact(artifact_path)
            ```
        """
        logged_artifact = self.experiment.get_artifact(artifact_path)
        artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
        logged_artifact.download(artifact_save_dir)

        metadata = logged_artifact.metadata
        data_dict = metadata.copy()
        data_dict["path"] = artifact_save_dir

        metadata_names = metadata.get("names")
        if isinstance(metadata_names, dict):
            data_dict["names"] = {int(k): v for k, v in metadata.get("names").items()}
        elif isinstance(metadata_names, list):
            data_dict["names"] = {int(k): v for k, v in zip(range(len(metadata_names)), metadata_names)}
        else:
            raise "Invalid 'names' field in dataset yaml file. Please use a list or dictionary"

        return self.update_data_paths(data_dict)

    def update_data_paths(self, data_dict):
        """
        Updates data paths in the dataset dictionary, defaulting 'path' to an empty string if not present.
        
        Args:
            data_dict (dict): The dataset configuration dictionary which contains keys like 'train', 'val', 'test', 
                              and an optional 'path'.
        
        Returns:
            dict: The updated dataset dictionary with modified paths for 'train', 'val', and 'test' splits.
        
        Notes:
            This function ensures that the paths for the dataset splits 'train', 'val', and 'test' are correctly 
            prefixed with the base 'path' provided in the data dictionary. If any of these splits are not present 
            or are incorrectly formatted, they are left unchanged.
            
        Examples:
            ```python
            data_dict = {
                'path': '/data',
                'train': 'images/train',
                'val': 'images/val'
            }
            new_data_dict = logger.update_data_paths(data_dict)
            # new_data_dict will be:
            # {
            #     'path': '/data',
            #     'train': '/data/images/train',
            #     'val': '/data/images/val'
            # }
            ```
        """
        path = data_dict.get("path", "")

        for split in ["train", "val", "test"]:
            if data_dict.get(split):
                split_path = data_dict.get(split)
                data_dict[split] = (
                    f"{path}/{split_path}" if isinstance(split, str) else [f"{path}/{x}" for x in split_path]
                )

        return data_dict

    def on_pretrain_routine_end(self, paths):
        """
        Called at the end of the pretraining routine to handle dataset artifact uploads and asset logging.
        
        Args:
            paths (list[str | Path]): List of paths to assets that need to be logged.
        
        Returns:
            None
        
        Notes:
            This method automatically skips execution if the training is being resumed or if dataset uploads are 
            disabled. If dataset uploads are enabled, this method will trigger the `upload_dataset_artifact` 
            method to upload the dataset as a Comet.ml artifact.
        """
        if self.opt.resume:
            return

        for path in paths:
            self.log_asset(str(path))

        if self.upload_dataset and not self.resume:
            self.upload_dataset_artifact()

        return

    def on_train_start(self):
        """
        Logs hyperparameters at the start of training.
        
        Args:
            None
        
        Returns:
            None
        
        Notes:
            This method is called once at the beginning of the training process to log the hyperparameters
            and any initial setup required for logging with Comet.ml.
        
            Please ensure that Comet.ml is properly configured in your environment. For more information
            on Comet.ml, visit: https://www.comet.com/
        
        Example:
            ```python
            comet_logger = CometLogger(opt, hyp)
            comet_logger.on_train_start()
            ```
        """
        self.log_parameters(self.hyp)

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        
        Logs the current epoch number and other relevant information related to the experiment.
        
        Args:
            None
        
        Returns:
            None
        
        Notes:
            This method is part of a callback mechanism that hooks into the training process. It helps
            log experiment metadata and other details to Comet.ml at the beginning of each epoch.
        """
        return

    def on_train_epoch_end(self, epoch):
        """
        Updates the current epoch in the experiment tracking at the end of each epoch.
        
        Args:
            epoch (int): The current training epoch.
        
        Returns:
            None (None)
        
        Note:
            This function is called internally by the training loop at the end of each epoch to appropriately 
            update the experiment's current epoch for logging purposes.
        """
        self.experiment.curr_epoch = epoch

        return

    def on_train_batch_start(self):
        """
        _train_batch_start()
            """Called at the start of each training batch.
        
            This method acts as a hook to perform operations at the beginning of each batch during the training phase.
            It can be used to log batch-level metrics or to execute other tasks that need to be done before processing
            each batch.
        
            Examples:
                ```python
                comet_logger = CometLogger(opt, hyp)
                comet_logger.on_train_batch_start()
                ```
        
            Note:
                This is often used in conjunction with other lifecycle hooks like on_epoch_start() and on_batch_end().
        """
        return

    def on_train_batch_end(self, log_dict, step):
        """
        Callback function that updates and logs metrics at the end of each training batch if conditions are met.
        
        Args:
            log_dict (dict): Dictionary containing the metrics to be logged. Keys are metric names, and values are 
                metric values.
            step (int): Current training step or batch index.
        
        Returns:
            None
        """
        self.experiment.curr_step = step
        if self.log_batch_metrics and (step % self.comet_log_batch_interval == 0):
            self.log_metrics(log_dict, step=step)

        return

    def on_train_end(self, files, save_dir, last, best, epoch, results):
        """
        Logs metadata and optionally saves model files at the end of training.
        
        Args:
          files (list[Path]): List of file paths to log as assets.
          save_dir (Path): Directory where training results and checkpoints are saved.
          last (Path): File path for the latest model checkpoint.
          best (Path): File path for the best model checkpoint.
          epoch (int): The current training epoch.
          results (dict): Training results including metrics.
        
        Returns:
          None
        
        Notes:
          If the `comet_log_predictions` flag is set, prediction metadata is logged to `image-metadata.json`.
        
        Example:
          ```python
          files = [Path("path/to/file1"), Path("path/to/file2")]
          save_dir = Path("path/to/save_dir")
          last_checkpoint = Path("path/to/last_checkpoint")
          best_checkpoint = Path("path/to/best_checkpoint")
          epoch_num = 10
          results = {"val_accuracy": 0.95}
        
          logger = CometLogger(opt, hyp)
          logger.on_train_end(files, save_dir, last_checkpoint, best_checkpoint, epoch_num, results)
          ```
        """
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
        if hasattr(self.opt, "comet_optimizer_id"):
            metric = results.get(self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_metric_value", metric)

        self.finish_run()

    def on_val_start(self):
        """
        Called at the start of validation, currently a placeholder with no functionality.
        
        Returns:
            None
        
        Notes:
            This function serves as a placeholder and does not carry out any operations at the moment.
            Future implementations may add functionality as needed.
        """
        return

    def on_val_batch_start(self):
        """
        Logs the start of a validation batch in the Comet experiment.
        
        Args:
            None
        
        Returns:
            None
        
        Notes:
            This function serves as a placeholder for future functionality related to batch-level validation logging in Comet. 
            Currently, it does not perform any operations.
            
        Example:
            ```python
            comet_logger = CometLogger(opt, hyp)
            comet_logger.on_val_batch_start()
            ```
        """
        return

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
        """
        Callback executed at the end of a validation batch, conditionally logs predictions to Comet ML.
        
        Args:
            batch_i (int): The index of the current batch within the validation epoch.
            images (torch.Tensor): The batch of images processed during validation.
            targets (torch.Tensor): The ground truth labels for the validation batch.
            paths (list[str]): List of file paths for the images in the validation batch.
            shapes (list[tuple]): List of original shapes for the images in the validation batch.
            outputs (torch.Tensor): The model's predictions for the validation batch.
        
        Returns:
            None
        """
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
        """
        Logs per-class metrics and confusion matrix to Comet.ml at the end of the validation phase.
        
        Args:
            nt (list[int]): Number of true instances for each class.
            tp (list[int]): Number of true positives for each class.
            fp (list[int]): Number of false positives for each class.
            p (list[float]): Precision for each class.
            r (list[float]): Recall for each class.
            f1 (list[float]): F1 score for each class.
            ap (list[float]): Average precision for each class.
            ap50 (list[float]): Average precision at IoU 0.5 for each class.
            ap_class (list[int]): List of class indices.
            confusion_matrix (object): Confusion matrix object containing the matrix data.
        
        Returns:
            None
        
        Note:
            This function will only log per-class metrics if `self.comet_log_per_class_metrics` is enabled and there are 
            multiple classes. The confusion matrix will be logged if `self.comet_log_confusion_matrix` is enabled.
        
        Example:
            ```python
            comet_logger.on_val_end(
                nt=[50, 45, 40],
                tp=[47, 40, 35],
                fp=[3, 5, 5],
                p=[0.94, 0.89, 0.87],
                r=[0.93, 0.88, 0.88],
                f1=[0.935, 0.885, 0.875],
                ap=[0.90, 0.85, 0.80],
                ap50=[0.95, 0.90, 0.85],
                ap_class=[0, 1, 2],
                confusion_matrix=confusion_matrix_object
            )
            ```
        """
        if self.comet_log_per_class_metrics and self.num_classes > 1:
            for i, c in enumerate(ap_class):
                class_name = self.class_names[c]
                self.experiment.log_metrics(
                    {
                        "mAP@.5": ap50[i],
                        "mAP@.5:.95": ap[i],
                        "precision": p[i],
                        "recall": r[i],
                        "f1": f1[i],
                        "true_positives": tp[i],
                        "false_positives": fp[i],
                        "support": nt[c],
                    },
                    prefix=class_name,
                )

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
                column_label="Actual Category",
                row_label="Predicted Category",
                file_name=f"confusion-matrix-epoch-{epoch}.json",
            )

    def on_fit_epoch_end(self, result, epoch):
        """
        Logs metrics at the end of each training epoch.
        
        Args:
            result (dict): A dictionary containing the metrics and results from the current training epoch.
            epoch (int): The current epoch count within the training process.
        
        Returns:
            None
        """
        self.log_metrics(result, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        """
        Callback to save model checkpoints periodically if conditions are met.
        
        Args:
            last (str): The path to the last saved model checkpoint.
            epoch (int): The current epoch number.
            final_epoch (bool): Flag indicating if this is the final epoch of training.
            best_fitness (float): The best fitness score achieved during training.
            fi (float): A fitness indicator based on the current training state.
        
        Returns:
            None
        
        Examples:
            ```python
            logger = CometLogger(opt, hyp, run_id="your-run-id")
            ...
            logger.on_model_save(last_model_path, current_epoch, is_final_epoch, best_fitness, fitness_indicator)
            ```
        
        Notes:
            This method is intended to be used as a callback function in the training loop to save model checkpoints 
            at defined intervals.
        """
        if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
            self.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_params_update(self, params):
        """
        Logs updated parameters during training.
        
        Args:
            params (dict[str, any]): Dictionary containing parameter names as keys and updated parameter values as values.
        
        Returns:
            None: This method does not return a value.
        
        Example:
            ```python
            comet_logger = CometLogger(opt, hyp)
            updated_params = {"learning_rate": 0.001, "batch_size": 32}
            comet_logger.on_params_update(updated_params)
            ```
        
        Note:
            This method is typically called automatically during the training process to ensure the latest parameter values are
            logged in Comet.ml.
        """
        self.log_parameters(params)

    def finish_run(self):
        """
        Ends the current experiment and logs its completion.
        
        Returns:
            None
        
        Raises:
            Exception: If there is an issue ending the experiment.
        
        Examples:
            ```python
            comet_logger = CometLogger(opt, hyp)
            comet_logger.finish_run()
            ```
        """
        self.experiment.end()
