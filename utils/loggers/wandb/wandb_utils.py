# Ultralytics YOLOv5 🚀, AGPL-3.0 license

# WARNING ⚠️ wandb is deprecated and will be removed in future release.
# See supported integrations at https://github.com/ultralytics/yolov5#integrations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from utils.general import LOGGER, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RANK = int(os.getenv("RANK", -1))
DEPRECATION_WARNING = (
    f"{colorstr('wandb')}: WARNING ⚠️ wandb is deprecated and will be removed in a future release. "
    f'See supported integrations at https://github.com/ultralytics/yolov5#integrations.'
)

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    LOGGER.warning(DEPRECATION_WARNING)
except (ImportError, AssertionError):
    wandb = None


class WandbLogger:
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information includes hyperparameters, system
    configuration and metrics, model metrics, and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets, models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type="Training"):
        """
        Initialize the WandbLogger.

        This logger primarily handles the initialization of logging to Weights & Biases (W&B) and prepares the required artifacts
        and metrics for logging during the training processes. It supports configuration for uploading datasets and resuming
        previous runs.

        Args:
            opt (namespace): Command line arguments for this run, containing various configuration settings.
            run_id (str | None): Optional run ID of a Weights & Biases run to be resumed. Default is None.
            job_type (str): Specifies the type of job for this run, defaults to 'Training'.

        Returns:
            None: This is an initializer method that sets up logging parameters and W&B configurations for the training process.

        Notes:
            - The wandb module is deprecated and will be removed in a future release.
            - Refer to the supported integrations at https://github.com/ultralytics/yolov5#integrations for more details.
        """
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, wandb.run if wandb else None
        self.val_artifact, self.train_artifact = None, None
        self.train_artifact_path, self.val_artifact_path = None, None
        self.result_artifact = None
        self.val_table, self.result_table = None, None
        self.max_imgs_to_log = 16
        self.data_dict = None
        if self.wandb:
            self.wandb_run = wandb.run or wandb.init(
                config=opt,
                resume="allow",
                project="YOLOv5" if opt.project == "runs/train" else Path(opt.project).stem,
                entity=opt.entity,
                name=opt.name if opt.name != "exp" else None,
                job_type=job_type,
                id=run_id,
                allow_val_change=True,
            )

        if self.wandb_run and self.job_type == "Training":
            if isinstance(opt.data, dict):
                # This means another dataset manager has already processed the dataset info (e.g. ClearML)
                # and they will have stored the already processed dict in opt.data
                self.data_dict = opt.data
            self.setup_training(opt)

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models.

        Args:
            opt (namespace): Command-line arguments for the current run. This includes hyperparameters, paths, and other
            configuration settings essential for setting up training.

        Returns:
            None

        Notes:
            - This function attempts to download model checkpoints and dataset artifacts if `opt.resume`
              starts with `WANDB_ARTIFACT_PREFIX`.
            - It updates the `data_dict` to include information from the previous run if resumed,
              and the paths of dataset artifacts if downloaded.
            - Initializes logging dictionary (`log_dict`) and bounding box interval (`bbox_interval`).
            - The function modifies `opt` attributes such as `weights`, `save_period`, `batch_size`,
              `bbox_interval`, `epochs`, `hyp`, and `imgsz` by extracting values from the W&B run config
              if a model artifact is downloaded.

        Examples:
            ```python
            from some_module import WandbLogger
            import argparse

            # Assume 'opt' is populated with necessary arguments
            parser = argparse.ArgumentParser()
            opt = parser.parse_args()

            logger = WandbLogger(opt)
            logger.setup_training(opt)
            ```

            For deprecation warnings and supported integrations, refer to:
            https://github.com/ultralytics/yolov5#integrations
        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            model_dir, _ = self.download_model_artifact(opt)
            if model_dir:
                self.weights = Path(model_dir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = (
                    str(self.weights),
                    config.save_period,
                    config.batch_size,
                    config.bbox_interval,
                    config.epochs,
                    config.hyp,
                    config.imgsz,
                )

        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # disable bbox_interval

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as a W&B artifact.

        Args:
            path (Path): Path of the directory containing the checkpoints.
            opt (namespace): Command-line arguments for the current run.
            epoch (int): Current epoch number.
            fitness_score (float): Fitness score for the current epoch.
            best_model (bool): Flag indicating if the current checkpoint is the best yet.

        Returns:
            None

        Example:
            ```python
            logger = WandbLogger(opt)
            logger.log_model(path=Path('/path/to/checkpoints'), opt=opt, epoch=5, fitness_score=0.95, best_model=True)
            ```
        """
        model_artifact = wandb.Artifact(
            f"run_{wandb.run.id}_model",
            type="model",
            metadata={
                "original_url": str(path),
                "epochs_trained": epoch + 1,
                "save period": opt.save_period,
                "project": opt.project,
                "total_epochs": opt.epochs,
                "fitness_score": fitness_score,
            },
        )
        model_artifact.add_file(str(path / "last.pt"), name="last.pt")
        wandb.log_artifact(
            model_artifact,
            aliases=[
                "latest",
                "last",
                f"epoch {str(self.current_epoch)}",
                "best" if best_model else "",
            ],
        )
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def val_one_image(self, pred, predn, path, names, im):
        """
        Evaluates model prediction for a single image, returning metrics and visualizations.

        Args:
            pred (torch.Tensor): The model predictions in standard format (x, y, w, h, obj_conf, class_conf).
            predn (torch.Tensor): The normalized model predictions (x, y, w, h, obj_conf, class_conf), typically used for
                                  evaluating performance against a standardized scale.
            path (str | Path): Path to the input image file being evaluated.
            names (List[str]): List of class names corresponding to the class indices used in the model predictions.
            im (numpy.ndarray): The input image in numpy array format.

        Returns:
            dict: A dictionary containing metrics and visualizations, including the processed image with predictions overlaid.

        Notes:
            - This method is a part of the deprecated `wandb` logging functionality, which will be removed in future releases.
            - Refer to supported integrations at https://github.com/ultralytics/yolov5#integrations for alternative logging options.
        """
        pass

    def log(self, log_dict):
        """
        Save the metrics to the logging dictionary.

        Args:
            log_dict (dict): Dictionary containing metrics or media to be logged in the current step.

        Returns:
            None

        Note:
            Weights & Biases (wandb) integration is deprecated and will be removed in future releases.
            See supported integrations at https://github.com/ultralytics/yolov5#integrations
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self):
        """
        Commit the log_dict, model artifacts, and Tables to W&B and flush the log_dict.

        Args:
            None

        Returns:
            None

        Raises:
            None

        Notes:
            This function logs the current state of self.log_dict to W&B. In case of an exception during logging,
            the W&B run is terminated, and training proceeds uninterrupted. The `wandb` library is deprecated
            and will be removed in future releases. See supported integrations at
            https://github.com/ultralytics/yolov5#integrations.
        """
        if self.wandb_run:
            with all_logging_disabled():
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(
                        f"An error occurred in wandb logger. The training will proceed without interruption. More info\n{e}"
                    )
                    self.wandb_run.finish()
                    self.wandb_run = None
                self.log_dict = {}

    def finish_run(self):
        """
        Log metrics if any and finish the current W&B run.

        Parameters:
            None

        Returns:
            None

        Notes:
            This method finalizes the logging process by logging any remaining metrics and closing the W&B run. This is
            especially useful to ensure all logs are sent before the script terminates. The function also disables all
            logging temporarily to avoid any conflicting logging statements.

        Examples:
            ```python
            # Initialize the WandbLogger
            wandb_logger = WandbLogger(opt)

            # Perform various logging operations
            wandb_logger.log({"accuracy": 0.95})

            # Finish the W&B run at the end of the script
            wandb_logger.finish_run()
            ```
        """
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()
            LOGGER.warning(DEPRECATION_WARNING)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    Disable all logging messages within the context.

    Args:
        highest_level (int): The highest logging level to disable. Defaults to logging.CRITICAL. This value can be set higher if custom logging levels are used.

    Yields:
        None

    Notes:
        This context manager can be useful to temporarily suppress logging output in libraries or modules, especially when you want to prevent logging messages from cluttering the console or logs.

    Example:
        ```python
        with all_logging_disabled():
            # Code that may produce logging output
            some_function_that_logs()
        # Logging resumes here
        ```
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
