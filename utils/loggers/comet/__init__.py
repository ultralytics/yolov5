import os
from importlib.metadata import metadata

try:
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    comet_ml = None

COMET_MODE = os.getenv("COMET_MODE", "online")
RANK = int(os.getenv("RANK", -1))


def _check_rank(fn, kwargs):
    if RANK != 0:
        return

    return fn(**kwargs)


class CometLogger:
    """Log metrics, parameters, source code, models and much more
    with Comet
    """

    def __init__(
        self, opt, run_id=None, job_type="Training", **experiment_kwargs
    ) -> None:
        self.job_type = job_type
        self.data_dict = None

        self.experiment_kwargs = experiment_kwargs
        if comet_ml is not None:
            self.experiment = self._get_experiment(COMET_MODE, run_id)

        if self.experiment is not None:
            self.experiment.log_other("Created from", "YOLOv5")
            self.experiment.log_parameters(vars(opt))

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self.experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                workspace=self.workspace,
                project_name=self.project_name,
                **self.experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self.experiment_kwargs,
                )

            return comet_ml.Experiment(
                workspace=self.workspace,
                project_name=self.project_name,
                **self.experiment_kwargs,
            )

        return

    def log(self, log_dict):
        self.experiment.log_metrics(log_dict)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        model_metadata = {
            "fitness_score": fitness_score,
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }
        self.experiment.log_model(path, metadata=model_metadata)

    def finish_run(self):
        if self.experiment is not None:
            self.experiment.end()
