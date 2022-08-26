import argparse
import json
import logging
import os
import sys
from pathlib import Path

import comet_ml

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import parse_opt, train
from utils.callbacks import Callbacks
from utils.general import increment_path
from utils.torch_utils import select_device

# Project Configuration
config = comet_ml.config.get_config()
COMET_PROJECT_NAME = config.get_string(
    os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5"
)


def get_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path"
    )
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="train, val image size (pixels)",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument(
        "--noval", action="store_true", help="only validate final epoch"
    )
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable AutoAnchor"
    )
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument(
        "--evolve",
        type=int,
        nargs="?",
        const=300,
        help="evolve hyperparameters for x generations",
    )
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help='--cache images in "ram" (default) or "disk"',
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/train", help="save to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "AdamW"],
        default="SGD",
        help="optimizer",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="EarlyStopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        type=int,
        default=[0],
        help="Freeze layers: backbone=10, first3=0 1 2",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (disabled if < 1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Automatic DDP Multi-GPU argument, do not modify",
    )
    # Comet Arguments
    parser.add_argument(
        "--comet_mode",
        type=str,
        help="Comet: Set whether to run Comet in online or offline mode.",
    )
    parser.add_argument(
        "--comet_save_model",
        action="store_true",
        help="Comet: Set to save model checkpoints.",
    )
    parser.add_argument(
        "--comet_model_name", type=str, help="Comet: Set the name for the saved model."
    )
    parser.add_argument(
        "--comet_overwrite_checkpoints",
        action="store_true",
        help="Comet: Overwrite existing model checkpoints.",
    )
    parser.add_argument(
        "--comet_checkpoint_filename",
        nargs="?",
        type=str,
        default="best.pt",
        help=(
            "Comet: Name of the checkpoint file to save to Comet."
            "Set to 'all' to log all checkpoints."
        ),
    )
    parser.add_argument(
        "--comet_log_batch_metrics",
        action="store_true",
        help="Comet: Set to log batch level training metrics.",
    )
    parser.add_argument(
        "--comet_log_batch_interval",
        type=int,
        default=1,
        help="Comet: Logging frequency for batch level training metrics.",
    )
    parser.add_argument(
        "--comet_log_prediction_interval",
        type=int,
        default=1,
        help=("Comet: How often to log predictions." "Applied at batch level."),
    )
    parser.add_argument(
        "--comet_log_confusion_matrix",
        action="store_true",
        help="Comet: Log a Confusion Matrix for the validation dataset.",
    )
    parser.add_argument(
        "--comet_log_predictions",
        action="store_true",
        help="Comet: Log Predictions on Images from the Validation Set",
    )
    parser.add_argument(
        "--comet_max_image_uploads",
        type=int,
        default=100,
        help="Comet: Maximum number of images to log to Comet.",
    )
    parser.add_argument(
        "--comet_upload_dataset",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Comet: Upload Dataset to Comet as an Artifact."
            "Set to 'train', 'val' or 'test' to upload a single dataset."
        ),
    )
    parser.add_argument(
        "--comet_artifact",
        type=str,
        help="Comet: Name of the Comet dataset Artifact to download.",
    )
    parser.add_argument(
        "--comet_optimizer_config",
        type=str,
        help="Comet: Path to a Comet Optimizer Config File.",
    )
    parser.add_argument(
        "--comet_optimizer_id",
        type=str,
        help="Comet: ID of the Comet Optimizer sweep.",
    )
    parser.add_argument(
        "--comet_optimizer_objective",
        type=str,
        help="Comet: Set to 'minimize' or 'maximize'.",
    )
    parser.add_argument(
        "--comet_optimizer_metric",
        type=str,
        help="Comet: Metric to Optimize.",
    )
    parser.add_argument(
        "--comet_optimizer_workers",
        type=int,
        default=1,
        help="Comet: Number of Parallel Workers to use with the Comet Optimizer.",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(parameters, opt):
    hyp_dict = {
        k: v for k, v in parameters.items() if k not in ["epochs", "batch_size"]
    }

    opt.save_dir = str(
        increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve
        )
    )
    opt.batch_size = parameters.get("batch_size")
    opt.epochs = parameters.get("epochs")

    device = select_device(opt.device, batch_size=opt.batch_size)
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    opt = get_args(known=True)

    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.project = str(opt.project)

    optimizer_id = os.getenv("COMET_OPTIMIZER_ID")
    if optimizer_id is None:
        with open(opt.comet_optimizer_config, "r") as f:
            optimizer_config = json.load(f)
        optimizer = comet_ml.Optimizer(optimizer_config)
    else:
        optimizer = comet_ml.Optimizer(optimizer_id)

    opt.comet_optimizer_id = optimizer.id
    status = optimizer.status()

    opt.comet_optimizer_objective = status["spec"]["objective"]
    opt.comet_optimizer_metric = status["spec"]["metric"]

    logger.info("COMET INFO: Starting Hyperparameter Sweep")
    for parameter in optimizer.get_parameters():
        run(parameter["parameters"], opt)
