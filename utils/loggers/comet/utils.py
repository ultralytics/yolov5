import os

import comet_ml
import yaml

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")


def download_model_checkpoint(opt, experiment):
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = opt.comet_model_name if opt.comet_model_name else COMET_MODEL_NAME
    model_asset_list = sorted(
        experiment.get_model_asset_list(model_name),
        key=lambda x: x["step"],
        reverse=True,
    )
    latest_model = model_asset_list[0]

    asset_id = latest_model["assetId"]
    model_filename = latest_model["fileName"]

    model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
    model_download_path = f"{model_dir}/{model_filename}"
    with open(model_download_path, "wb") as f:
        f.write(model_binary)

    opt.weights = model_download_path


def set_opt_parameters(opt, experiment):
    """Update the opts Namespace with parameters
    from Comet's ExistingExperiment when resuming a run

    Args:
        opt (argparse.Namespace): _description_
        experiment (comet_ml.APIExperiment): _description_
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


def check_comet_weights(opt):
    """Downloads model weights from Comet and updates the
    weights path to point to saved weights location

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        bool: _description_
    """

    if isinstance(opt.weights, str):
        if opt.weights.startswith(COMET_PREFIX):
            api = comet_ml.API()
            experiment_path = opt.weights.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            download_model_checkpoint(opt, experiment)
            return True

    return None


def check_comet_resume(opt):
    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            api = comet_ml.API()
            experiment_path = opt.resume.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            set_opt_parameters(opt, experiment)
            download_model_checkpoint(opt, experiment)

            return True

    return None
