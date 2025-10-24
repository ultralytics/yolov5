# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
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
    Format `value` for JSON serialization (e.g. unwrap tensors).

    Fall back to strings.
    """
    if isinstance(value, torch.Tensor):
        try:
            value = value.item()
        except ValueError:  # "only one element tensors can be converted to Python scalars"
            pass
    return value if isinstance(value, float) else str(value)


class Loggers:
    """Initializes and manages various logging utilities for tracking YOLOv5 training and validation metrics."""

    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        """Initializes loggers for YOLOv5 training and validation metrics, paths, and options."""
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
        """Fetches dataset dictionary from remote logging services like ClearML, Weights & Biases, or Comet ML."""
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict

        return data_dict

    def on_train_start(self):
        """Initializes the training process for Comet ML logger if it's configured."""
        if self.comet_logger:
            self.comet_logger.on_train_start()

    def on_pretrain_routine_start(self):
        """Invokes pre-training routine start hook for Comet ML logger if available."""
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_start()

    def on_pretrain_routine_end(self, labels, names):
        """Callback that runs at the end of pre-training routine, logging label plots if enabled."""
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
        """Logs training batch end events, plots images, and updates external loggers with batch-end data."""
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
        """Callback that updates the current epoch in Weights & Biases at the end of a training epoch."""
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

        if self.comet_logger:
            self.comet_logger.on_train_epoch_end(epoch)

    def on_val_start(self):
        """Callback that signals the start of a validation phase to the Comet logger."""
        if self.comet_logger:
            self.comet_logger.on_val_start()

    def on_val_image_end(self, pred, predn, path, names, im):
        """Callback that logs a validation image and its predictions to WandB or ClearML."""
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)
        if self.clearml:
            self.clearml.log_image_with_boxes(path, pred, names, im)

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        """Logs validation batch results to Comet ML during training at the end of each validation batch."""
        if self.comet_logger:
            self.comet_logger.on_val_batch_end(batch_i, im, targets, paths, shapes, out)

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        """Logs validation results to WandB or ClearML at the end of the validation process."""
        if self.wandb or self.clearml:
            files = sorted(self.save_dir.glob("val*.jpg"))
        if self.wandb:
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
        if self.clearml:
            self.clearml.log_debug_samples(files, title="Validation")

        if self.comet_logger:
            self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """Callback that logs metrics and saves them to CSV or NDJSON at the end of each fit (train+val) epoch."""
        x = dict(zip(self.keys, vals))
        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = "" if file.exists() else (("%20s," * n % tuple(["epoch", *self.keys])).rstrip(",") + "\n")  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch, *vals])).rstrip(",") + "\n")
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
                best_results = [epoch, *vals[3:7]]
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
        """Callback that handles model saving events, logging to Weights & Biases or ClearML if enabled."""
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
        """Callback that runs at the end of training to save plots and log results."""
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
        """Updates experiment hyperparameters or configurations in WandB, Comet, or ClearML."""
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
        if self.comet_logger:
            self.comet_logger.on_params_update(params)
        if self.clearml:
            self.clearml.task.connect(params)


class GenericLogger:
    """
    YOLOv5 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...).

    Arguments:
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, include=("tb", "wandb", "clearml")):
        """Initializes a generic logger with optional TensorBoard, W&B, and ClearML support."""
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
        """Logs metrics to CSV, TensorBoard, W&B, and ClearML; `metrics` is a dict, `epoch` is an int."""
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch", *keys])).rstrip(",") + "\n")  # header
            with open(self.csv, "a") as f:
                f.write(s + ("%23.5g," * n % tuple([epoch, *vals])).rstrip(",") + "\n")

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(metrics, step=epoch)

        if self.clearml:
            self.clearml.log_scalars(metrics, epoch)

    def log_images(self, files, name="Images", epoch=0):
        """Logs images to all loggers with optional naming and epoch specification."""
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
        """Logs model graph to all configured loggers with specified input image size."""
        if self.tb:
            log_tensorboard_graph(self.tb, model, imgsz)

    def log_model(self, model_path, epoch=0, metadata=None):
        """Logs the model to all configured loggers with optional epoch and metadata."""
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
        """Updates logged parameters in WandB and/or ClearML if enabled."""
        if self.wandb:
            wandb.run.config.update(params, allow_val_change=True)
        if self.clearml:
            self.clearml.task.connect(params)


def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    """Logs the model graph to TensorBoard with specified image size and model."""
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
    """Converts a local project name to a standardized web project name with optional suffixes."""
    if not project.startswith("runs/train"):
        return project
    suffix = "-Classify" if project.endswith("-cls") else "-Segment" if project.endswith("-seg") else ""
    return f"YOLOv5{suffix}"
