"""Utilities and tools for tracking runs with Mlflow."""
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from utils.general import LOGGER

try:
    import mlflow

    assert hasattr(mlflow, "__version__")
except (ImportError, AssertionError):
    mlflow = None



class MlflowLogger(object):
    def __init__(self, opt, run_id=None) -> None:
        try:
            LOGGER.info(f"mlflow run_id from init {run_id}")
            self.mlflow, self.mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run(run_id=run_id)
            if self.mlflow_active_run is not None:
                self.run_id = self.mlflow_active_run.info.run_id
                LOGGER.info(f"from mlflow logger: {self.run_id}")
        except Exception as err:
            LOGGER.error(f"Mlflow couldn't create or find experiment: {str(err)}")
            self.mlflow = None
            self.mlflow_active_run = None
            self.current_epoch = 0

    def log_artifacts(self, last, epoch=None):
        LOGGER.info(f"from mlflow log_artifacts: {last, epoch}")

    def log_params(self, params: dict):
        if self.mlflow is not None:
            self.mlflow.log_params(params=params)

    def log_metrics(self, metrics: dict, epoch: int = None):
        if self.mlflow is not None:
            self.mlflow.log_metrics(metrics=metrics, step=epoch)

    def finish_run(self):
        if self.mlflow is not None and self.mlflow_active_run is not None:
            self.mlflow.end_run()
