import math

from sparsezoo import Zoo
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

from utils.torch_utils import is_parallel


def _get_model_framework_file(model, path):
    transfer_request = 'recipe_type=transfer' in path
    checkpoint_available = any([file.checkpoint for file in model.framework_files])
    final_available = any([not file.checkpoint for file in model.framework_files])

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if avaiable and requested
        return [file for file in model.framework_files if file.checkpoint][0]
    elif final_available:
        # default to returning final state, if available
        return [file for file in model.framework_files if not file.checkpoint][0]

    raise ValueError(f"Could not find a valid framework file for {path}")


def check_download_sparsezoo_weights(path):
    if isinstance(path, str):
        if path.startswith("zoo:"):
            # load model from the SparseZoo and override the path with the new download
            model = Zoo.load_model_from_stub(path)
            file = _get_model_framework_file(model, path)
            path = file.downloaded_path()

        return path

    if isinstance(path, list):
        return [check_download_sparsezoo_weights(p) for p in path]

    return path


class SparseMLWrapper(object):
    def __init__(self, model, recipe):
        self.enabled = bool(recipe)
        self.model = model.module if is_parallel(model) else model
        self.recipe = recipe
        self.manager = ScheduledModifierManager.from_yaml(recipe) if self.enabled else None
        self.logger = None

    def state_dict(self):
        return {
            'recipe': str(self.manager) if self.enabled else None,
        }

    def apply(self):
        if not self.enabled:
            return

        self.manager.apply(self.model)

    def initialize(self, start_epoch):
        if not self.enabled:
            return

        self.manager.initialize(self.model, start_epoch)

    def initialize_loggers(self, logger, tb_writer, wandb_logger, rank):
        self.logger = logger

        if not self.enabled or rank not in [-1, 0]:
            return

        def _logging_lambda(log_tag, log_val, log_vals, step, walltime):
            if not wandb_logger or not wandb_logger.wandb:
                return

            if log_val is not None:
                wandb_logger.log({log_tag: log_val})

            if log_vals:
                wandb_logger.log(log_vals)

        self.manager.initialize_loggers([
            SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                tensorboard=tb_writer,
            )
        ])

        if wandb_logger.wandb:
            artifact = wandb_logger.wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb_logger.wandb.log_artifact(artifact)

    def modify(self, scaler, optimizer, model, dataloader):
        if not self.enabled:
            return scaler

        return self.manager.modify(model, optimizer, steps_per_epoch=len(dataloader), wrap_optim=scaler)

    def check_lr_override(self, scheduler):
        # Override lr scheduler if recipe makes any LR updates
        if self.enabled and self.manager.learning_rate_modifiers:
            self.logger.info('Disabling LR scheduler, managing LR using SparseML recipe')
            scheduler = None

        return scheduler

    def check_epoch_override(self, epochs):
        # Override num epochs if recipe explicitly modifies epoch range
        if self.enabled and self.manager.epoch_modifiers and self.manager.max_epochs:
            epochs = self.manager.max_epochs or epochs  # override num_epochs
            self.logger.info(f'Overriding number of epochs from SparseML manager to {epochs}')

        return epochs

    def qat_active(self, epoch):
        if not self.enabled or not self.manager.quantization_modifiers:
            return False

        qat_start = min([mod.start_epoch for mod in self.manager.quantization_modifiers])

        return qat_start < epoch + 1

    def reset_best(self, epoch):
        if not self.enabled:
            return False

        # if pruning is active or quantization just started, need to reset best checkpoint
        # this is in case the pruned and/or quantized model do not fully recover
        pruning_start = math.floor(max([mod.start_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        pruning_end = math.ceil(max([mod.end_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        qat_start = math.floor(max([mod.start_epoch for mod in self.manager.quantization_modifiers])) \
            if self.manager.quantization_modifiers else -1

        return (pruning_start <= epoch <= pruning_end) or epoch == qat_start
