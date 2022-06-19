"""Utilities and tools for tracking runs with Mlflow."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Union

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import LOGGER, colorstr

try:
    import mlflow
    assert hasattr(mlflow, "__version__")
except (ImportError, AssertionError):
    mlflow = None


class MlflowLogger:
    """Log training run, artifacts, parameters, and metrics to Mlflow.

    This logger expects that Mlflow is setup by the user.
    """

    def __init__(self, opt: Namespace) -> None:
        """Initializes the MlflowLogger

        Args:
            opt (Namespace): Commandline arguments for this run
        """
        prefix = colorstr("Mlflow: ")
        try:
            self.mlflow, self.mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run()
            if self.mlflow_active_run is not None:
                self.run_id = self.mlflow_active_run.info.run_id
                LOGGER.info(f"{prefix}Using run_id({self.run_id})")
                self.log_params(vars(opt))
                self.log_metrics(vars(opt), is_param=True)
        except Exception as err:
            LOGGER.error(f"{prefix}Failing init - {str(err)}")
            LOGGER.warning(f"{prefix}Continuining without Mlflow")
            self.mlflow = None
            self.mlflow_active_run = None

    def log_artifacts(self, artifact: Path, epoch: int = None) -> None:
        """Member function to log artifacts (either directory or single item).

        If the artifacts need to be saved as checkpoints, set the epoch.

        Args:
            artifact (Path): File or folder to be logged
            epoch (int, optional): Epoch during training. Set value to save input as a checkpoint. Defaults to None.
        """
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        artifact_name = artifact.stem
        if artifact.is_dir():
            name = f"{artifact_name}_epoch{str(epoch).zfill(4)}" if epoch is not None else artifact_name
            self.mlflow.log_artifacts(str(f"{artifact.resolve()}/"), artifact_path=name)
        else:
            name = f"{artifact_name}_epoch{str(epoch).zfill(4)}{artifact.suffix}" if epoch is not None else None
            self.mlflow.log_artifact(artifact.resolve(), artifact_path=name)

    def log_model(self, model) -> None:
        """Member function to log model as an Mlflow model.

        Args:
            model (nn.Module): The pytorch model
        """
        self.mlflow.pytorch.log_model(model, code_paths=[ROOT.resolve()])

    def log_params(self, params: Dict[str, Any]) -> None:
        """Member funtion to log parameters.
        Mlflow doesn't have mutable parameters and so this function is used
        only to log initial training parameters.

        Args:
            params (Dict[str, Any]): Parameters as dict
        """
        self.mlflow.log_params(params=params)

    def log_metrics(self, metrics: Dict[str, float], epoch: int = None, is_param: bool = False) -> None:
        """Member function to log metrics.
        Mlflow requires metrics to be floats.

        Args:
            metrics (Dict[str, float]): Dictionary with metric names and values
            epoch (int, optional): Training epoch. Defaults to None.
            is_param (bool, optional): Set it to True to log keys with a prefix "params/". Defaults to False.
        """
        prefix = "param/" if is_param else ""
        metrics_dict = {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, Union[float, int])}
        self.mlflow.log_metrics(metrics=metrics_dict, step=epoch)

    def finish_run(self) -> None:
        """Member function to end mlflow run.
        """
        self.mlflow.end_run()
