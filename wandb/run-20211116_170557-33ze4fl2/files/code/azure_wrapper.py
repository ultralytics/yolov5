from pathlib import Path
import yaml
import os
from pprint import pprint
import subprocess
import typer

# Code in submitted run
try:
    from azureml.core import Run
except ModuleNotFoundError:
    pass


def _find_module_wheel_path(module_name):
    module_name = module_name.replace("-", "_")
    module_wheel_paths = list(Path(".").rglob(f"**/{module_name}*.whl"))
    if len(module_wheel_paths) == 1:
        return module_wheel_paths[0]
    elif len(module_wheel_paths) == 0:
        raise Exception(f"Cannot find wheel associated with package: {module_name}")
    else:
        raise Exception(
            f"Found several wheels associated with package: {module_name} ({module_wheel_paths})"
        )


def _setup():
    # Setting up W&B
    if (
        os.getenv("AZUREML_ARM_RESOURCEGROUP") is not None
    ):  # checking if we are in Azure, unless you have really weird local env variables
        run = Run.get_context()
        secret_value = run.get_secret(name="WANDB-BOT-API-KEY")  # Secret called
        # WANDB-API-KEY2 was created in Azure KeyVault
        os.environ["WANDB_API_KEY"] = secret_value

    # Install aisa utils
    try:
        import aisa_utils
    except ModuleNotFoundError:
        aisa_utils_wheel_path = _find_module_wheel_path("aisa_utils")
        subprocess.run(["pip", "install", f"{aisa_utils_wheel_path}"])


def _get_data_yaml(dataset_location: Path, add_test: bool = False) -> dict:
    dataset_data_yaml_path = list(dataset_location.rglob("*data.yaml"))
    if len(dataset_data_yaml_path) == 1:
        dataset_data_yaml_path = dataset_data_yaml_path[0]
        with dataset_data_yaml_path.open("r") as file:
            data_yaml = yaml.load(file)
    elif len(dataset_data_yaml_path) > 1:
        raise Exception(
            f"Multiple data.yaml files found at {dataset_location}: {dataset_data_yaml_path}"
        )
    else:
        data_yaml = dict(
            nc=1,
            names=["Sperm"],
        )
    print(dataset_location)
    print(dataset_data_yaml_path)
    # Overwrite location keys to be able to launch from anywhere
    data_yaml["path"] = str(dataset_data_yaml_path.parent.as_posix())
    data_yaml["train"] = "images/train"
    data_yaml["val"] = "images/val"
    if add_test:
        data_yaml["test"] = "images/test"
    else:
        data_yaml.pop("test", None)

    return data_yaml


_setup()
# CLI app
app = typer.Typer(add_completion=True)


@app.command()
def train(
    yolo_model_version: str = typer.Argument(
        ...,
        help="Model Name to train (ex: yolov5s)",
    ),
    train_dataset_location: Path = typer.Argument(
        ...,
        help="Location of train dataset (yaml or path to root).",
    ),
    test_dataset_location: Path = typer.Argument(
        ...,
        help="Location of test dataset (yaml or path to root).",
    ),
    test_video_dataset_location: Path = typer.Option(
        None,
        help="Location of test video dataset (root of folder with videos).",
    ),
    batch_size: int = typer.Option(
        16,
        help="Size of batches to use for training.",
    ),
    image_size: int = typer.Option(
        960,
        help="Input image size.",
    ),
    epochs: int = typer.Option(
        200,
        help="Number of epochs to train for.",
    ),
    hyp: Path = typer.Option(
        Path("data/hyps/hyp.scratch.yaml"),
        help="Path to hyp file.",
    ),
    project: str = typer.Option(
        "test_training_results",
        help="WandB project name to upload to.",
    ),
):
    import train
    import mojo_test

    # Some values shared between train/test scripts
    entity = "mojo-ia"

    #### TRAINING CODE ####
    # Create data.yaml is a root path is given (hard code extra values for now).
    if train_dataset_location.is_dir():
        train_data = _get_data_yaml(train_dataset_location)
        pprint(train_data)
        train_yaml_file_path = Path("data.yaml")
        with train_yaml_file_path.open("w") as file:
            yaml.dump(train_data, file)
        print(f"Created data yaml at {train_yaml_file_path} containing:")
    elif train_dataset_location.is_file() and train_dataset_location.suffix == ".yaml":
        train_yaml_file_path = train_dataset_location
    else:
        raise Exception(f"{train_dataset_location} not supported as an dataset type.")

    print("Running training function...")
    if os.name == "nt":
        workers = 1
    else:
        workers = 4
    path_to_best_model = train.run(
        cfg=f"models/{yolo_model_version}.yaml",
        weights=f"{yolo_model_version}.pt",
        data=f"{train_yaml_file_path}",
        hyp=f"{hyp}",
        project=project,
        name=f"{yolo_model_version}-{image_size}-{hyp.stem}",
        epochs=epochs,
        batch_size=batch_size,
        imgsz=image_size,
        workers=workers,
        entity=entity,
        patience=100,
        cache="disk"
    )
    print("Finished training function...")
    #### END OF TRAINING CODE ####

    #### TESTING ####

    mojo_test_data = _get_data_yaml(test_dataset_location, add_test=True)
    mojo_test_yaml_file_path = Path("test_data.yaml")
    with mojo_test_yaml_file_path.open("w") as file:
        yaml.dump(mojo_test_data, file)
    print(f"Created data yaml at {mojo_test_yaml_file_path} containing:")

    print("Running mojo testing function...")
    mojo_test.mojo_test(
        mojo_test_yaml_file_path,
        [path_to_best_model],
        batch_size=batch_size,
        imgsz=image_size,
        project=project,
        name=f"{yolo_model_version}-{image_size}",
        entity=entity,
        test_video_root=test_video_dataset_location,
    )


def cli():
    app()


if __name__ == "__main__":
    app()
