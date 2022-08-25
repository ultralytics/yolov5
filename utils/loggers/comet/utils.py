import logging
import os

try:
    import comet_ml
except (ModuleNotFoundError, ImportError) as e:
    comet_ml = None

import yaml

logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")


def download_model_checkpoint(opt, experiment):
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = opt.comet_model_name if opt.comet_model_name else COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)

    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {model_name}")
        return

    model_asset_list = sorted(
        model_asset_list,
        key=lambda x: x["step"],
        reverse=True,
    )

    checkpoint_filename = opt.comet_checkpoint_filename
    logged_checkpoint_map = {
        asset["fileName"]: asset["assetId"] for asset in model_asset_list
    }
    asset_id = logged_checkpoint_map.get(checkpoint_filename)

    try:
        if asset_id is None:
            # Fetch latest checkpoint
            asset_id = model_asset_list[0]["assetId"]
            asset_filename = model_asset_list[0]["fileName"]
            logger.info(
                f"COMET INFO: Checkpoint {checkpoint_filename} not found."
                f"Defaulting to latest checkpoint {asset_filename}"
            )

        else:
            asset_filename = checkpoint_filename

        model_binary = experiment.get_asset(
            asset_id, return_type="binary", stream=False
        )
        model_download_path = f"{model_dir}/{asset_filename}"
        with open(model_download_path, "wb") as f:
            f.write(model_binary)

        opt.weights = model_download_path

    except Exception as e:
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def set_opt_parameters(opt, experiment):
    """Update the opts Namespace with parameters
    from Comet's ExistingExperiment when resuming a run

    Args:
        opt (argparse.Namespace): Namespace of command line options
        experiment (comet_ml.APIExperiment): Comet API Experiment object
    """
    asset_list = experiment.get_asset_list()
    resume_string = opt.resume
    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            asset_id = asset["assetId"]
            asset_binary = experiment.get_asset(
                asset_id, return_type="binary", stream=False
            )
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                setattr(opt, key, value)
            opt.resume = resume_string

    # Save hyperparamers to YAML file
    # Necessary to pass checks in training script
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    hyp_yaml_path = f"{save_dir}/hyp.yaml"
    with open(hyp_yaml_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_yaml_path


def check_comet_weights(opt):
    """Downloads model weights from Comet and updates the
    weights path to point to saved weights location

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if weights are successfully downloaded
            else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.weights, str):
        if opt.weights.startswith(COMET_PREFIX):
            api = comet_ml.API()
            experiment_path = opt.weights.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            download_model_checkpoint(opt, experiment)
            return True

    return None


def check_comet_resume(opt):
    """Restores run parameters to its original state based on the model checkpoint
    and logged Experiment parameters.

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if the run is restored successfully
            else return None
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            api = comet_ml.API()
            experiment_path = opt.resume.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            set_opt_parameters(opt, experiment)
            download_model_checkpoint(opt, experiment)

            return True

    return None
