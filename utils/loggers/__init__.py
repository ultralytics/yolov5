# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Logging utils."""

import json
import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch

from utils.general import LOGGER, colorstr, cv2
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv("RANK", -1))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:

    def SummaryWriter(*args):
        """Fall back to SummaryWriter returning None if TensorBoard is not installed."""
        return None  # None = SummaryWriter(str)


try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.2") and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    import clearml

    assert hasattr(clearml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None

try:
    if RANK in {0, -1}:
        import comet_ml

        assert hasattr(comet_ml, "__version__")  # verify package import not local dir
        from utils.loggers.comet import CometLogger

    else:
        comet_ml = None
except (ImportError, AssertionError):
    comet_ml = None


def _json_default(value):
    """
    Format `value` for JSON serialization, handling special types like PyTorch tensors.

    Args:
        value (Any): The input value to be formatted for JSON serialization. It can be of any type.

    Returns:
        str | Any: A JSON-serializable representation of the input value. If the value is a PyTorch tensor, it attempts
        to convert it to a scalar. For any other types, it attempts to return the input value directly.

    Examples:
        ```python
        import torch

        tensor = torch.tensor(5)
        serialized_tensor = _json_default(tensor)
        # serialized_tensor would be 5

        complex_tensor = torch.tensor([1, 2, 3])
        serialized_complex_tensor = _json_default(complex_tensor)
        # serialized_complex_tensor would be [1, 2, 3]
        ```

    Notes:
        If the value is a PyTorch tensor and could not be converted to a scalar, it falls back to returning the tensor
        as is.
    """
    if isinstance(value, torch.Tensor):
        try:
            value = value.item()
        except ValueError:  # "only one element tensors can be converted to Python scalars"
            pass
    return value if isinstance(value, float) else str(value)


class Loggers:
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        """
        Initializes loggers for YOLOv5 training and validation metrics, paths, and options.

        Args:
            save_dir (str | Path | None): Directory to save training logs and checkpoints. Default is None.
            weights (str | None): Path to pre-trained weights to be used. Default is None.
            opt (argparse.Namespace | None): Command-line arguments in a namespace object containing various options. Default is None.
            hyp (dict | None): Dictionary of hyperparameters. Default is None.
            logger (logging.Logger | None): Logger instance for printing results to console. Default is None.
            include (tuple[str] | None): List of loggers to include. Possible values are "csv", "tb", "wandb", "clearml", "comet".
                                       Defaults to a predefined set of loggers ("csv", "tb", "wandb", "clearml", "comet").

        Returns:
            None

        Notes:
            - Ensure `comet_ml` package is installed to utilize Comet logging features. Run `pip install comet_ml` for installation.
            - TensorBoard logging can be started with the command `tensorboard --logdir {save_dir.parent}` and viewed at http://localhost:6006/
            - WandB logging requires the `wandb` package and login with `wandb.login()`.
            - ClearML logging requires the `clearml` package and proper configuration as per https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme
        """
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",  # metrics
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        self.best_keys = ["best/epoch", "best/precision", "best/recall", "best/mAP_0.5", "best/mAP_0.5:0.95"]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv
        self.ndjson_console = "ndjson_console" in self.include  # log ndjson to console
        self.ndjson_file = "ndjson_file" in self.include  # log ndjson to file

        # Messages
        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet"
            self.logger.info(s)
        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and "wandb" in self.include:
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt)
        else:
            self.wandb = None

        # ClearML
        if clearml and "clearml" in self.include:
            try:
                self.clearml = ClearmlLogger(self.opt, self.hyp)
            except Exception:
                self.clearml = None
                prefix = colorstr("ClearML: ")
                LOGGER.warning(
                    f"{prefix}WARNING ⚠️ ClearML is installed but not configured, skipping ClearML logging."
                    f" See https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme"
                )

        else:
            self.clearml = None

        # Comet
        if comet_ml and "comet" in self.include:
            if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                run_id = self.opt.resume.split("/")[-1]
                self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)

            else:
                self.comet_logger = CometLogger(self.opt, self.hyp)

        else:
            self.comet_logger = None

    @property
    def remote_dataset(self):
        """
        Fetches dataset dictionary from remote logging services like ClearML, Weights & Biases, or Comet ML.

        Returns:
            dict | None: Dictionary containing dataset information, if available from any remote logging service;
            otherwise, returns None.

        Notes:
            This property checks the availability of dataset dictionaries from ClearML, Weights & Biases, and Comet ML
            in that order. The first non-null dictionary encountered is returned.

        Example:
            ```python
            loggers = Loggers(include=("clearml", "wandb", "comet"))
            dataset_info = loggers.remote_dataset
            ```
        """
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict

        return data_dict

    def on_train_start(self):
        """
        Initializes the training process for Comet ML logger and other configured loggers.

        Returns:
            None

        Notes:
            This function is a part of the setup phase in the YOLOv5 training pipeline and ensures that all necessary logging
            services are appropriately started and configured prior to commencing the training loop. This may involve creating
            new logger instances or initializing existing ones, depending on the configuration.
        """
        if self.comet_logger:
            self.comet_logger.on_train_start()

    def on_pretrain_routine_start(self):
        """
        Invokes pre-training routine start hook for Comet ML logger if available.

        Returns:
            None

        Notes:
            This method is intended to be called before the start of the training routine to ensure that
            any configured logging services are properly initialized and ready to log the training metrics.

            Currently, this supports Comet ML logger initialization, but the condition can be expanded
            to include other loggers in future updates.
        """
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_start()

    def on_pretrain_routine_end(self, labels, names):
        """
        Callback that runs at the end of pre-training routine, logging label plots if enabled.

        Args:
          labels (list): List of label data for the dataset.
          names (list): List of class names corresponding to the labels.

        Returns:
          None

        Notes:
          This function generates and saves label plots to the specified save directory. If logging integrations
          like Weights & Biases or Comet ML are enabled, it logs the generated plots to their respective platforms.
        """
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob("*labels*.jpg")  # training labels
            if self.wandb:
                self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
            if self.comet_logger:
                self.comet_logger.on_pretrain_routine_end(paths)
            if self.clearml:
                for path in paths:
                    self.clearml.log_plot(title=path.stem, plot_path=path)

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        """
        Logs training batch end events, plots images, and updates external loggers with batch-end data.

        Args:
            model (torch.nn.Module): The model being trained.
            ni (int): Number of integrated batches since the training started.
            imgs (torch.Tensor): Batch of input images.
            targets (torch.Tensor): Ground truth targets for the batch.
            paths (list[str]): File paths of the input images.
            vals (list[float]): List of loss values [box_loss, obj_loss, cls_loss].

        Returns:
            None

        Notes:
            - This function logs the end of a training batch, and if specific conditions are satisfied (e.g., a certain
              number of batches have been processed), it generates image plots and logs them to TensorBoard, Weights & Biases,
              or ClearML, depending on the configuration.
            - Plots are generated for the first few batches to track initial training behavior and after several batches
              for monitoring ongoing training progress.
            - When TensorBoard is enabled and sync batch normalization is not used, the model graph is also logged.

        Examples:
            ```python
            loggers = Loggers(save_dir="runs/train/exp")
            loggers.on_train_batch_end(model, 1, imgs, targets, paths, [0.5, 0.3, 0.2])
            ```
        """
        log_dict = dict(zip(self.keys[:3], vals))
        # Callback runs on train batch end
        # ni: number integrated batches (since train start)
        if self.plots:
            if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and not self.opt.sync_bn:
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10 and (self.wandb or self.clearml):
                files = sorted(self.save_dir.glob("train*.jpg"))
                if self.wandb:
                    self.wandb.log({"Mosaics": [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})
                if self.clearml:
                    self.clearml.log_debug_samples(files, title="Mosaics")

        if self.comet_logger:
            self.comet_logger.on_train_batch_end(log_dict, step=ni)

    def on_train_epoch_end(self, epoch):
        """
        Callback that updates loggers and performs relevant actions at the end of a training epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None

        Notes:
            - If Weights & Biases (W&B) logger is enabled, it updates the current epoch count.
            - Ensure that W&B, Comet ML, and ClearML logging services are properly configured before invoking this method if these loggers are included.
        """
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

        if self.comet_logger:
            self.comet_logger.on_train_epoch_end(epoch)

    def on_val_start(self):
        """
        Callback that signals the start of a validation phase to all configured loggers.

        Returns:
            None

        Notes:
            This method initializes or prepares any configured loggers to handle the beginning of the validation phase.
            Specifically, it engages the Comet ML logger, if available, to start tracking validation metrics and other
            relevant information.
        """
        if self.comet_logger:
            self.comet_logger.on_val_start()

    def on_val_image_end(self, pred, predn, path, names, im):
        """
        Logs a validation image and its predictions at the end of the validation phase.

        Args:
            pred (torch.Tensor): The predicted bounding boxes for the validation image.
            predn (torch.Tensor): The normalized predicted bounding boxes for the validation image.
            path (str | Path): The file path of the validation image.
            names (list of str): A list of class names corresponding to the predicted labels.
            im (numpy.ndarray): The original validation image.

        Returns:
            None

        Notes:
            This function logs validation images and their corresponding predictions to Weights & Biases (WandB) and ClearML
            for further analysis and visualization. Ensure that the WandB and ClearML integrations are correctly initialized
            and configured before calling this method.
        """
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)
        if self.clearml:
            self.clearml.log_image_with_boxes(path, pred, names, im)

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        """
        Logs validation batch results to Comet ML during training at the end of each validation batch.

        Args:
            batch_i (int): Index of the current batch.
            im (torch.Tensor): Tensor of input images in the batch.
            targets (torch.Tensor): Tensor of target labels for the images.
            paths (list[str]): List of file paths for the images.
            shapes (torch.Tensor): Tensor of the original shapes of the images.
            out (torch.Tensor): Tensor of model outputs for the current batch.

        Returns:
            None

        Notes:
            This function logs validation batch results to Comet ML if a CometLogger is configured.
        """
        if self.comet_logger:
            self.comet_logger.on_val_batch_end(batch_i, im, targets, paths, shapes, out)

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        """
        Logs validation results to WandB or ClearML at the end of the validation process.

        Args:
          nt (torch.Tensor): Tensor containing the number of true instances for each class.
          tp (torch.Tensor): Tensor containing true positive counts for each class.
          fp (torch.Tensor): Tensor containing false positive counts for each class.
          p (torch.Tensor): Tensor containing precision values for each class.
          r (torch.Tensor): Tensor containing recall values for each class.
          f1 (torch.Tensor): Tensor containing F1 scores for each class.
          ap (torch.Tensor): Tensor containing average precision (AP) values for each class.
          ap50 (torch.Tensor): Tensor containing AP values at a 0.5 IoU threshold for each class.
          ap_class (torch.Tensor): Tensor containing class indices corresponding to the provided AP values.
          confusion_matrix (torch.Tensor): Tensor representing the confusion matrix.

        Returns:
          None

        Examples:
          ```python
          loggers = Loggers(...)  # Initialize Loggers
          loggers.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)  # Log validation end results
          ```
        """
        if self.wandb or self.clearml:
            files = sorted(self.save_dir.glob("val*.jpg"))
        if self.wandb:
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
        if self.clearml:
            self.clearml.log_debug_samples(files, title="Validation")

        if self.comet_logger:
            self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """
        Logs metrics and saves them to CSV or NDJSON at the end of each fit (train+val) epoch.

        Args:
            vals (list[float]): List of metric values to log.
            epoch (int): The current epoch number.
            best_fitness (float): Current value of the best fitness metric.
            fi (float): Fitness value used to determine the best model.

        Returns:
            None

        Notes:
            This method logs the metrics to different loggers (`csv`, `tb`, `wandb`, `clearml`) based on the configuration set
            during initialization of the Loggers class. Metrics are also saved to a JSON or NDJSON format if specified.

        Examples:
            ```python
            logger.on_fit_epoch_end(vals=[0.5, 0.3, 0.2, 0.8], epoch=10, best_fitness=0.8, fi=0.85)
            ```
        """
        x = dict(zip(self.keys, vals))
        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = "" if file.exists() else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")
        if self.ndjson_console or self.ndjson_file:
            json_data = json.dumps(dict(epoch=epoch, **x), default=_json_default)
        if self.ndjson_console:
            print(json_data)
        if self.ndjson_file:
            file = self.save_dir / "results.ndjson"
            with open(file, "a") as f:
                print(json_data, file=f)

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        elif self.clearml:  # log to ClearML if TensorBoard not used
            self.clearml.log_scalars(x, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[i]  # log best results in the summary
            self.wandb.log(x)
            self.wandb.end_epoch()

        if self.clearml:
            self.clearml.current_epoch_logged_images = set()  # reset epoch image limit
            self.clearml.current_epoch += 1

        if self.comet_logger:
            self.comet_logger.on_fit_epoch_end(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        """
        Callback that handles model saving events, logging to Weights & Biases, ClearML, or Comet ML if configured.

        Args:
            last (Path | str): Path to the last saved model checkpoint.
            epoch (int): Current epoch number.
            final_epoch (bool): Indicator to identify if the current epoch is the final one.
            best_fitness (float): Metric representing the best fitness value achieved so far.
            fi (float): Fitness score of the current model after the epoch.

        Returns:
            None

        Notes:
            This function interacts with integrated logging services such as Weights & Biases, ClearML, and Comet ML
            to save model artifacts and update the respective dashboards with model checkpoints periodically or at the end
            of training.

        Example:
            ```python
            last_checkpoint_path = Path("/checkpoints/last.pt")
            current_epoch = 23
            is_final_epoch = False
            best_fitness_score = 0.85
            current_fitness_score = 0.82

            loggers.on_model_save(last_checkpoint_path, current_epoch, is_final_epoch, best_fitness_score, current_fitness_score)
            ```

        Links:
            - For more details on ClearML logging, visit: https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration#readme
        """
        if (epoch + 1) % self.opt.save_period == 0 and not final_epoch and self.opt.save_period != -1:
            if self.wandb:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
            if self.clearml:
                self.clearml.task.update_output_model(
                    model_path=str(last), model_name="Latest Model", auto_delete_file=False
                )

        if self.comet_logger:
            self.comet_logger.on_model_save(last, epoch, final_epoch, best_fitness, fi)

    def on_train_end(self, last, best, epoch, results):
        """
        Callback that runs at the end of training to save plots and log results.

        Args:
            last (Path): Path to the last model checkpoint.
            best (Path): Path to the best model checkpoint.
            epoch (int): The final epoch number in the training.
            results (list[float]): A list of performance metrics from the training process.

        Returns:
            None

        Notes:
            - Saves various training result plots (e.g., F1 curve, PR curve, etc.) to the save directory.
            - Logs final metrics and plots to supported loggers (TensorBoard, Weights & Biases, ClearML, Comet ML).
            - Ensures that relevant images and plots are correctly recorded and saved for further analysis.
        """
        if self.plots:
            plot_results(file=self.save_dir / "results.csv")  # save results.png
        files = ["results.png", "confusion_matrix.png", *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R"))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb and not self.clearml:  # These images are already captured by ClearML by now, we don't want doubles
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC")

        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(
                    str(best if best.exists() else last),
                    type="model",
                    name=f"run_{self.wandb.wandb_run.id}_model",
                    aliases=["latest", "best", "stripped"],
                )
            self.wandb.finish_run()

        if self.clearml and not self.opt.evolve:
            self.clearml.log_summary(dict(zip(self.keys[3:10], results)))
            [self.clearml.log_plot(title=f.stem, plot_path=f) for f in files]
            self.clearml.log_model(
                str(best if best.exists() else last), "Best Model" if best.exists() else "Last Model", epoch
            )

        if self.comet_logger:
            final_results = dict(zip(self.keys[3:10], results))
            self.comet_logger.on_train_end(files, self.save_dir, last, best, epoch, final_results)

    def on_params_update(self, params: dict):
        """
        Updates experiment hyperparameters or configurations in WandB, Comet, or ClearML.

        Args:
            params (dict): Dictionary containing the hyperparameters or configuration updates.

        Returns:
            None

        Notes:
            - This method allows dynamic updates to the experiment configuration parameters for integrated logging platforms.
            - It supports updating configurations on WandB, Comet, and ClearML loggers if they are enabled.

        Examples:
            ```python
            # Create a loggers object
            loggers = Loggers(save_dir='path/to/save', opt=opt, hyp=hyp, include=['wandb', 'comet'])

            # Update hyperparameters during the experiment
            new_params = {'learning_rate': 0.001, 'momentum': 0.9}
            loggers.on_params_update(new_params)
            ```
        """
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
        if self.comet_logger:
            self.comet_logger.on_params_update(params)
        if self.clearml:
            self.clearml.task.connect(params)


class GenericLogger:
    """
    YOLOv5 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, include=("tb", "wandb", "clearml")):
        """
        Initializes a generic logger with support for TensorBoard, Weights & Biases (W&B), and ClearML logging.

        Args:
            opt (object): The run configuration and hyperparameters.
            console_logger (logging.Logger): Logger instance for console output.
            include (tuple[str]): A tuple of loggers to include for logging. Defaults to ("tb", "wandb", "clearml").

        Returns:
            None

        Examples:
            ```python
            from utils.loggers import GenericLogger

            opt = {...}  # some configuration object
            console_logger = LOGGER  # assuming LOGGER is a configured logger instance
            logger = GenericLogger(opt, console_logger)
            ```

        Notes:
            - Ensure that the required dependencies like TensorBoard, wandb, and clearml are installed and correctly configured.
            - Configure the `opt` object appropriately to ensure logging is performed correctly.
        """
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / "results.csv"  # CSV logger
        if "tb" in self.include:
            prefix = colorstr("TensorBoard: ")
            self.console_logger.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(self.save_dir))

        if wandb and "wandb" in self.include:
            self.wandb = wandb.init(
                project=web_project_name(str(opt.project)), name=None if opt.name == "exp" else opt.name, config=opt
            )
        else:
            self.wandb = None

        if clearml and "clearml" in self.include:
            try:
                # Hyp is not available in classification mode
                hyp = {} if "hyp" not in opt else opt.hyp
                self.clearml = ClearmlLogger(opt, hyp)
            except Exception:
                self.clearml = None
                prefix = colorstr("ClearML: ")
                LOGGER.warning(
                    f"{prefix}WARNING ⚠️ ClearML is installed but not configured, skipping ClearML logging."
                    f" See https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration"
                )
        else:
            self.clearml = None

    def log_metrics(self, metrics, epoch):
        """
        Logs metrics to CSV, TensorBoard, Weights & Biases (W&B), and ClearML.

        Args:
            metrics (dict): The dictionary of metrics to log. Keys should be metric names (str), and values should be metric values (float).
            epoch (int): The current epoch number.

        Returns:
            None

        Notes:
            - The function appends metrics to a CSV file located at `self.save_dir / "results.csv"`.
            - If TensorBoard logging is enabled, metrics are logged using `self.tb.add_scalar`.
            - If Weights & Biases (W&B) logging is enabled, metrics are logged using `self.wandb.log`.
            - Metrics are also logged to ClearML if enabled, using the appropriate ClearML logging utilities.

        Example:
            ```python
            logger = GenericLogger(opt, console_logger, include=("tb", "wandb", "clearml"))
            metrics = {'train/loss': 0.123, 'val/accuracy': 0.987}
            logger.log_metrics(metrics, epoch=1)
            ```
        """
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
            with open(self.csv, "a") as f:
                f.write(s + ("%23.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(metrics, step=epoch)

        if self.clearml:
            self.clearml.log_scalars(metrics, epoch)

    def log_images(self, files, name="Images", epoch=0):
        """
        Logs images to TensorBoard, Weights & Biases (W&B), and ClearML with optional naming and epoch specification.

        Args:
            files (Path | str | list[Path | str]): Image files to log.
            name (str): Optional. The name to use for logging images. Defaults to 'Images'.
            epoch (int): Optional. The epoch number to associate with these images. Defaults to 0.

        Returns:
            None

        Example:
            ```python
            logger = GenericLogger(opt, console_logger, include=["tb", "wandb", "clearml"])
            image_files = ["path/to/image1.jpg", "path/to/image2.jpg"]
            logger.log_images(image_files, name="Sample Images", epoch=5)
            ```

        Notes:
            - Ensure the files exist before passing them to this function.
            - The `name` parameter helps categorize or identify the images in the log.
            - The function supports logging to multiple services simultaneously as defined in the `include` attribute.
            - The `epoch` number helps in organizing the images logged during different epochs of model training.
        """
        files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC")

        if self.wandb:
            self.wandb.log({name: [wandb.Image(str(f), caption=f.name) for f in files]}, step=epoch)

        if self.clearml:
            if name == "Results":
                [self.clearml.log_plot(f.stem, f) for f in files]
            else:
                self.clearml.log_debug_samples(files, title=name)

    def log_graph(self, model, imgsz=(640, 640)):
        """
        Logs model graph to all configured loggers with specified input image size.

        Args:
            model (torch.nn.Module): The PyTorch model to log the graph for.
            imgsz (tuple[int, int]): Input image size as a tuple. Defaults to (640, 640).

        Returns:
            None

        Examples:
            ```python
            logger = GenericLogger(opt, console_logger)
            logger.log_graph(model, imgsz=(640, 480))
            ```
        """
        if self.tb:
            log_tensorboard_graph(self.tb, model, imgsz)

    def log_model(self, model_path, epoch=0, metadata=None):
        """
        Logs the model to all configured loggers with optional epoch and metadata.

        Args:
            model_path (str | Path): Path to the model file to be logged.
            epoch (int, optional): The current epoch number. Defaults to 0.
            metadata (dict, optional): Additional metadata to log with the model. Defaults to None.

        Returns:
            None

        Note:
            Ensure that the model file exists at the given path before invoking this method.

        Example:
            ```python
            logger = GenericLogger(opt, console_logger, include=("wandb", "clearml"))
            model_path = "path/to/model.pt"
            logger.log_model(model_path, epoch=10, metadata={"accuracy": 0.95})
            ```
        """
        if metadata is None:
            metadata = {}
        # Log model to all loggers
        if self.wandb:
            art = wandb.Artifact(name=f"run_{wandb.run.id}_model", type="model", metadata=metadata)
            art.add_file(str(model_path))
            wandb.log_artifact(art)
        if self.clearml:
            self.clearml.log_model(model_path=model_path, model_name=model_path.stem)

    def update_params(self, params):
        """
        Updates hyperparameters or configurations in WandB and/or ClearML if enabled.

        Args:
          params (dict): The parameters to be updated in the respective loggers.

        Returns:
          None

        Notes:
          This method enables dynamic updating of experiment parameters during runtime, facilitating better experiment tracking and logging.
        """
        if self.wandb:
            wandb.run.config.update(params, allow_val_change=True)
        if self.clearml:
            self.clearml.task.connect(params)


def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    """
    log_tensorboard_graph(tb, model, imgsz=(640, 640)) Logs the model graph to TensorBoard with a specified image size.

    Args:
        tb (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard SummaryWriter instance used for logging.
        model (torch.nn.Module): The model to log.
        imgsz (int | tuple[int, int], optional): Size of the input images. Default is (640, 640).

    Returns:
        None

    Notes:
        This function utilizes the `torch.jit.trace` to generate the computational graph and logs it to TensorBoard.
        Ensure the input image tensor is filled with zeros to avoid unexpected values during tracing.
    """
    try:
        p = next(model.parameters())  # for device, type
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # expand
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)  # input image (WARNING: must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress jit trace warning
            tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")


def web_project_name(project):
    """
    Converts a local project name to a standardized web project name with optional suffixes.

    Args:
        project (str): The local project name.

    Returns:
        str: The standardized web project name with relevant suffix if applicable.

    Examples:
        ```python
        web_project_name("runs/train123")  # returns "runs/train123"
        web_project_name("runs/train123-seg")  # returns "runs/train123-Segment"
        ```
    Notes:
        This function checks if the project name starts with "runs/train" and appends relevant suffixes for classification (`"-Classify"`) or segmentation (`"-Segment"`) projects.
    """
    if not project.startswith("runs/train"):
        return project
    suffix = "-Classify" if project.endswith("-cls") else "-Segment" if project.endswith("-seg") else ""
    return f"YOLOv5{suffix}"
