import os

import comet_ml

COMET_MODE = os.getenv("COMET_MODE", "online")
COMET_SAVE_MODEL = os.getenv("COMET_SAVE_MODEL", "false").lower() == "true"
COMET_OVERWRITE_CHECKPOINTS = (
    os.getenv("COMET_OVERWRITE_CHECKPOINTS", "true").lower() == "true"
)
COMET_LOG_BATCH_METRICS = (
    os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
)
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
RANK = int(os.getenv("RANK", -1))


class CometLogger:
    """Log metrics, parameters, source code, models and much more
    with Comet
    """

    def __init__(
        self, opt, run_id=None, job_type="Training", **experiment_kwargs
    ) -> None:
        self.job_type = job_type
        self.opt = opt

        # Flags
        self.comet_mode = self.opt.comet_mode if self.opt.comet_mode else COMET_MODE

        self.log_batch_metrics = (
            opt.comet_log_batch_metrics
            if opt.comet_log_batch_metrics
            else COMET_LOG_BATCH_METRICS
        )
        self.save_model = (
            opt.comet_save_model if opt.comet_save_model else COMET_SAVE_MODEL
        )
        self.overwrite_checkpoints = (
            opt.comet_overwrite_checkpoints
            if opt.comet_overwrite_checkpoints
            else COMET_OVERWRITE_CHECKPOINTS
        )
        self.batch_logging_interval = (
            opt.comet_log_batch_interval
            if opt.comet_log_batch_interval
            else COMET_BATCH_LOGGING_INTERVAL
        )

        # Default parameters to pass to Experiment objects
        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            # for ddp logging set output logging simple
        }
        self.default_experiment_kwargs.update(experiment_kwargs)
        self.experiment = self._get_experiment(self.comet_mode, run_id)

        if self.experiment is not None:
            if run_id is None:
                self.log_parameters(vars(opt))
                self.log_asset(opt.hyp, metadata={"type": "hyp-config-file"})
                self.log_asset(
                    f"{opt.save_dir}/opt.yaml", metadata={"type": "opt-config-file"}
                )
                self.experiment.log_other("Created from", "YOLOv5")
                self.experiment.log_other(
                    "Run ID",
                    f"{self.experiment.workspace}/{self.experiment.project_name}/{self.experiment.id}",
                )

    def _get_experiment(self, mode, experiment_id=None):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                **self.default_experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    **self.default_experiment_kwargs,
                )

            return comet_ml.Experiment(
                **self.default_experiment_kwargs,
            )

        return

    def log_metrics(self, log_dict, **kwargs):
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        self.experiment.log_asset(asset_path, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        if not self.save_model:
            return

        model_metadata = {
            "fitness_score": fitness_score,
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }

        if opt.comet_checkpoint_filename is "all":
            model_path = str(path)
        else:
            model_path = str(path) + f"/{opt.comet_checkpoint_filename}"

        self.experiment.log_model(
            "yolov5",
            model_path,
            metadata=model_metadata,
            overwrite=self.overwrite_checkpoints,
        )

    def finish_run(self):
        if self.experiment is not None:
            self.experiment.end()
