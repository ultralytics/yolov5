# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from export import run
from utils.general import LOGGER, colorstr, cv2, emojis
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel

RANK = int(os.getenv("RANK", -1))

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.2") and RANK in {
        0,
        -1,
    }:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    assert RANK in {0, -1}
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # verify package import not local dir
    comet_ml.init()
    from utils.loggers.comet import CometLogger

except (ImportError, AssertionError):
    comet_ml = None

# Make this configurable?
LOGGERS = (
    "csv",
    "tb",
    "wandb",
    "clearml",
    "comet",
)  # text-file, TensorBoard, Weights & Biases


class Loggers:
    # YOLOv5 Loggers class
    def __init__(
        self,
        save_dir=None,
        weights=None,
        opt=None,
        hyp=None,
        logger=None,
        include=LOGGERS,
    ):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
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
        self.best_keys = [
            "best/epoch",
            "best/precision",
            "best/recall",
            "best/mAP_0.5",
            "best/mAP_0.5:0.95",
        ]
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr("Weights & Biases: ")
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            self.logger.info(emojis(s))

        # TensorBoard
        s = self.save_dir
        if "tb" in self.include and not self.opt.evolve:
            prefix = colorstr("TensorBoard: ")
            self.logger.info(
                f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and "wandb" in self.include:
            wandb_artifact_resume = isinstance(
                self.opt.resume, str
            ) and self.opt.resume.startswith("wandb-artifact://")
            run_id = (
                torch.load(self.weights).get("wandb_id")
                if self.opt.resume and not wandb_artifact_resume
                else None
            )
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
            # temp warn. because nested artifacts not supported after 0.12.10
            if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.11"):
                self.logger.warning(
                    "YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected."
                )
        else:
            self.wandb = None

        if "comet" in self.include:
            try:
                if isinstance(self.opt.resume, str) and self.opt.resume.startswith(
                    "comet://"
                ):
                    run_id = self.opt.resume.split("/")[-1]
                    self.comet_logger = CometLogger(self.opt, run_id=run_id)

                else:
                    self.comet_logger = CometLogger(self.opt)

            except Exception as e:
                self.comet_logger = None
                raise (e)

    def on_train_start(self):
        if self.comet_logger:
            self.comet_logger.log_parameters(self.hyp)

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob("*labels*.jpg")  # training labels
        if self.wandb:
            self.wandb.log(
                {"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]}
            )

        if self.comet_logger:
            for x in paths:
                self.comet_logger.log_asset(str(x))

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, vals):
        log_dict = dict(zip(self.keys[0:3], vals))
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                if (
                    self.tb and not self.opt.sync_bn
                ):  # --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # suppress jit trace warning
                        self.tb.add_graph(
                            torch.jit.trace(
                                de_parallel(model), imgs[0:1], strict=False
                            ),
                            [],
                        )
            if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"  # filename
                plot_images(imgs, targets, paths, f)
            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob("train*.jpg"))
                self.wandb.log(
                    {
                        "Mosaics": [
                            wandb.Image(str(f), caption=f.name)
                            for f in files
                            if f.exists()
                        ]
                    }
                )

        if self.comet_logger:
            if self.comet_logger.log_batch_metrics and (
                ni % self.comet_logger.batch_logging_interval == 0
            ):
                self.comet_logger.log_metrics(log_dict, step=ni)

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = dict(zip(self.keys, vals))

        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1  # number of cols
            s = (
                ""
                if file.exists()
                else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")
            )  # add header
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[
                        i
                    ]  # log best results in the summary
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

        if self.comet_logger:
            self.comet_logger.log_metrics(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if (
                (epoch + 1) % self.opt.save_period == 0 and not final_epoch
            ) and self.opt.save_period != -1:
                self.wandb.log_model(
                    last.parent, self.opt, epoch, fi, best_model=best_fitness == fi
                )

        if self.comet_logger:
            if (
                (epoch + 1) % self.opt.save_period == 0 and not final_epoch
            ) and self.opt.save_period != -1:
                self.comet_logger.log_model(
                    last.parent, self.opt, epoch, fi, best_model=best_fitness == fi
                )

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        if plots:
            plot_results(file=self.save_dir / "results.csv")  # save results.png
        files = [
            "results.png",
            "confusion_matrix.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [
            (self.save_dir / f) for f in files if (self.save_dir / f).exists()
        ]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb:
            for f in files:
                self.tb.add_image(
                    f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC"
                )
        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            self.wandb.log(
                {"Results": [wandb.Image(str(f), caption=f.name) for f in files]}
            )
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(
                    str(best if best.exists() else last),
                    type="model",
                    name=f"run_{self.wandb.wandb_run.id}_model",
                    aliases=["latest", "best", "stripped"],
                )
            self.wandb.finish_run()

        if self.comet_logger:
            for f in files:
                self.comet_logger.log_asset(f, metadata={"epoch": epoch})
            self.comet_logger.log_asset(
                f"{self.save_dir}/results.csv", metadata={"epoch": epoch}
            )

            self.comet_logger.finish_run()

    def on_params_update(self, params):
        # Update hyperparams or configs of the experiment
        # params: A dict containing {param: value} pairs
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
