# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import os
from urllib.parse import urlparse

try:
    import comet_ml
except ImportError:
    comet_ml = None

import yaml

logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def download_model_checkpoint(opt, experiment):
    """Downloads YOLOv5 model checkpoint from Comet ML experiment, updating `opt.weights` with download path."""
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)

    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {model_name}")
        return

    model_asset_list = sorted(
        model_asset_list,
        key=lambda x: x["step"],
        reverse=True,
    )
    logged_checkpoint_map = {asset["fileName"]: asset["assetId"] for asset in model_asset_list}

    resource_url = urlparse(opt.weights)
    checkpoint_filename = resource_url.query

    if checkpoint_filename:
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME

    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment")
        return

    try:
        logger.info(f"COMET INFO: Downloading checkpoint {checkpoint_filename}")
        asset_filename = checkpoint_filename

        model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
        model_download_path = f"{model_dir}/{asset_filename}"
        with open(model_download_path, "wb") as f:
            f.write(model_binary)

        opt.weights = model_download_path

    except Exception as e:
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def set_opt_parameters(opt, experiment):
    """Update the opts Namespace with parameters from Comet's ExistingExperiment when resuming a run.

    Args:
        opt (argparse.Namespace): Namespace of command line options
        experiment (comet_ml.APIExperiment): Comet API Experiment object
    """
    asset_list = experiment.get_asset_list()
    resume_string = opt.resume

    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            asset_id = asset["assetId"]
            asset_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string

    # Save hyperparameters to YAML file
    # Necessary to pass checks in training script
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    hyp_yaml_path = f"{save_dir}/hyp.yaml"
    with open(hyp_yaml_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path


def check_comet_weights(opt):
    """Downloads model weights from Comet and updates the weights path to point to saved weights location.

    Args:
        opt (argparse.Namespace): Command Line arguments passed to YOLOv5 training script

    Returns:
        None/bool: Return True if weights are successfully downloaded else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.weights, str) and opt.weights.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.weights)
        experiment_path = f"{resource.netloc}{resource.path}"
        experiment = api.get(experiment_path)
        download_model_checkpoint(opt, experiment)
        return True

    return None


def check_comet_resume(opt):
    """Restores run parameters to its original state based on the model checkpoint and logged Experiment parameters.

    Args:
        opt (argparse.Namespace): Command Line arguments passed to YOLOv5 training script

    Returns:
        None/bool: Return True if the run is restored successfully else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str) and opt.resume.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.resume)
        experiment_path = f"{resource.netloc}{resource.path}"
        experiment = api.get(experiment_path)
        set_opt_parameters(opt, experiment)
        download_model_checkpoint(opt, experiment)

        return True

    return None
