"""Utilities and tools for tracking runs with Mlflow."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

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

from collections.abc import MutableMapping


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
        if opt.weights is not None and str(opt.weights).strip() != "":
            model_name = Path(opt.weights).stem
        else:
            model_name = "yolov5"
        self.model_name = model_name
        self.weights = Path(opt.weights)
        self.client = mlflow.tracking.MlflowClient()
        self.log_params(vars(opt))
        self.log_metrics(vars(opt), is_param=True)

    @staticmethod
    def _flatten_params(params_dict, parent_key="", sep="/"):
        """Static method to flatten configs for logging to mlflow"""
        items = []
        for key, value in params_dict.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(MlflowLogger._flatten_params(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def log_artifacts(self, artifact: Path, relpath: str = None) -> None:
        """Member function to log artifacts (either directory or single item).

        Args:
            artifact (Path): File or folder to be logged
            relpath (str): Name (or path) relative to experiment for logging artifact in mlflow
        """
        if not isinstance(artifact, Path):
            artifact = Path(artifact)
        if artifact.is_dir():
            self.mlflow.log_artifacts(f"{artifact.resolve()}/", artifact_path=str(artifact.stem))
        else:
            self.mlflow.log_artifact(str(artifact.resolve()), artifact_path=relpath)

    def log_model(self, model_path: Path, model_name: str = None) -> None:
        """Member function to log model as an Mlflow model.

        Args:
            model_path: Path to the model .pt being logged
            model_name: Name (or path) relative to experiment for logging model in mlflow
        """
        self.mlflow.pyfunc.log_model(artifact_path=self.model_name if model_name is None else model_name,
                                     code_path=[str(ROOT.resolve())],
                                     artifacts={"model_path": str(model_path.resolve())},
                                     python_model=self.mlflow.pyfunc.PythonModel())

    def log_params(self, params: Dict[str, Any]) -> None:
        """Member funtion to log parameters.
        Mlflow doesn't have mutable parameters and so this function is used
        only to log initial training parameters.

        Args:
            params (Dict[str, Any]): Parameters as dict
        """
        try:
            flattened_params = MlflowLogger._flatten_params(params_dict=params)
            run = self.client.get_run(run_id=self.run_id)
            logged_params = run.data.params
            [
                self.mlflow.log_param(key=k, value=v) for k, v in flattened_params.items()
                if k not in logged_params and v is not None and str(v).strip() != ""]
        except Exception as err:
            LOGGER.warning(f"Mlflow: failed to log all params because - {err}")

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
            f"{prefix}{k.replace(':','-')}": float(v)
            for k, v in metrics.items() if (isinstance(v, float) or isinstance(v, int))}
        self.mlflow.log_metrics(metrics=metrics_dict, step=epoch)

    def finish_run(self) -> None:
        """Member function to end mlflow run.
        """
        self.mlflow.end_run()
