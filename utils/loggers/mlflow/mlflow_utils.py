"""Utilities and tools for tracking runs with Mlflow."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import torch

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
                self.setup(opt)
        except Exception as err:
            LOGGER.error(f"{prefix}Failing init - {repr(err)}")
            LOGGER.warning(f"{prefix}Continuining without Mlflow")
            self.mlflow = None
            self.mlflow_active_run = None

    def setup(self, opt: Namespace) -> None:
        self.model_name = Path(opt.weights).stem
        self.weights = Path(opt.weights)
        try:
            self.client = mlflow.tracking.MlflowClient()
            run = self.client.get_run(run_id=self.run_id)
            logged_params = run.data.params
            remaining_params = {k: v for k, v in vars(opt).items() if k not in logged_params}
            self.log_params(remaining_params)
        except Exception as err:
            LOGGER.warning(f"Mlflow: not logging params because - {err}")
        self.log_metrics(vars(opt), is_param=True)

    def log_artifacts(self, artifact: Path) -> None:
        """Member function to log artifacts (either directory or single item).

        Args:
            artifact (Path): File or folder to be logged
        """
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        if artifact.is_dir():
            self.mlflow.log_artifacts(f"{artifact.resolve()}/", artifact_path=str(artifact.stem))
        else:
            self.mlflow.log_artifact(artifact.resolve())

    def log_model(self, model_path) -> None:
        """Member function to log model as an Mlflow model.

        Args:
            model (nn.Module): The pytorch model
        """
        model = torch.hub.load(repo_or_dir=str(ROOT.resolve()), model="custom", path=str(model_path), source="local")
        if self.weights.exists() and (self.weights.parent.resolve() == ROOT.resolve()):
            self.weights.unlink()
        self.mlflow.pytorch.log_model(model, artifact_path=self.model_name, code_paths=[ROOT.resolve()])

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
        metrics_dict = {
            f"{prefix}{k}": float(v)
            for k, v in metrics.items() if (isinstance(v, float) or isinstance(v, int))}
        self.mlflow.log_metrics(metrics=metrics_dict, step=epoch)

    def finish_run(self) -> None:
        """Member function to end mlflow run.
        """
        self.mlflow.end_run()
