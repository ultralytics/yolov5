import logging
import math
import os
from typing import Optional

from sparsezoo import Zoo
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

from utils.torch_utils import is_parallel
import torch
import numpy

_LOGGER = logging.getLogger(__file__)
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
    def __init__(self, model, checkpoint_recipe, train_recipe, steps_per_epoch=-1, one_shot=False):
        self.enabled = bool(train_recipe)
        self.model = model.module if is_parallel(model) else model
        self.checkpoint_manager = ScheduledModifierManager.from_yaml(checkpoint_recipe) if checkpoint_recipe else None
        self.manager = ScheduledModifierManager.from_yaml(train_recipe) if train_recipe else None
        self.logger = None
        self.start_epoch = None
        self.steps_per_epoch = steps_per_epoch
        self.one_shot = one_shot
        self.train_recipe = train_recipe

        if self.one_shot:
            self._apply_one_shot()

    def state_dict(self):
        manager = (ScheduledModifierManager.compose_staged(self.checkpoint_manager, self.manager)
        if self.checkpoint_manager and self.enabled else self.manager)

        return {
            'recipe': str(manager) if self.enabled else None,
        }

    def apply_checkpoint_structure(self):
        if self.checkpoint_manager:
            self.checkpoint_manager.apply_structure(self.model, math.inf)

    def initialize(self, start_epoch):
        if not self.enabled:
            return
        self.manager.initialize(self.model, start_epoch)
        self.start_epoch = start_epoch

    def initialize_loggers(self, logger, tb_writer, wandb_logger):
        self.logger = logger

        if not self.enabled:
            return

        def _logging_lambda(tag, value, values, step, wall_time, level):
            if not wandb_logger or not wandb_logger.wandb:
                return

            if value is not None:
                wandb_logger.log({tag: value})

            if values:
                wandb_logger.log(values)

        self.manager.initialize_loggers([
            SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                tensorboard=tb_writer,
            )
        ])

        if wandb_logger and wandb_logger.wandb:
            artifact = wandb_logger.wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb_logger.wandb.log_artifact(artifact)

    def modify(self, scaler, optimizer, model, dataloader):
        if not self.enabled:
            return scaler

        self.steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch > 0 else len(dataloader)
        return self.manager.modify(model, optimizer, steps_per_epoch=self.steps_per_epoch, wrap_optim=scaler)

    def check_lr_override(self, scheduler, rank):
        # Override lr scheduler if recipe makes any LR updates
        if self.enabled and self.manager.learning_rate_modifiers:
            if rank in [0,-1]:
                self.logger.info('Disabling LR scheduler, managing LR using SparseML recipe')
            scheduler = None

        return scheduler

    def check_epoch_override(self, epochs, rank):
        # Override num epochs if recipe explicitly modifies epoch range
        if self.enabled and self.manager.epoch_modifiers and self.manager.max_epochs:
            if rank in [0,-1]:
                self.logger.info(f'Overriding number of epochs from SparseML manager to {epochs}')
            epochs = self.manager.max_epochs + self.start_epoch or epochs  # override num_epochs

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

    def _apply_one_shot(self):
        if self.manager is not None:
            self.manager.apply(self.model)
            _LOGGER.info(f"Applied recipe {self.train_recipe} in one-shot manner")
        else:
            _LOGGER.info(f"Training recipe for one-shot application not recognized by the manager. Got recipe: "
                         f"{self.train_recipe}"
                         )

    def save_sample_inputs_outputs(
        self,
        dataloader: "Dataloader",  # flake8 : noqa F8421
        num_export_samples=100,
        save_dir: Optional[str] = None,
    ):

        save_dir = save_dir or ""
        if not dataloader:
            raise ValueError(
                f"Expected a data loader for exporting samples. Got {dataloader}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        exported_samples = 0

        sample_in_dir = os.path.join(save_dir, "sample-inputs")
        sample_out_dir = os.path.join(save_dir, "sample-outputs")

        os.makedirs(sample_in_dir, exist_ok=True)
        os.makedirs(sample_out_dir, exist_ok=True)

        for _, (images, _, _, _) in enumerate(dataloader):
            images = (
                images.float() / 255
            )  # uint8 to float32, 0-255 to 0.0-1.0

            device_imgs = images.to(device, non_blocking=True)
            outs = self.model(device_imgs)

            # Move to cpu for exporting
            ins = images.detach().to("cpu")
            outs = [elem.detach().to("cpu") for elem in outs]
            outs_gen = zip(*outs)

            for sample_in, sample_out in zip(ins, outs_gen):
                sample_out = list(sample_out)
                file_idx = f"{exported_samples}".zfill(4)

                sample_input_filename = os.path.join(f"{sample_in_dir}", f"inp-{file_idx}.npz")
                numpy.savez(sample_input_filename, sample_in)

                sample_output_filename = os.path.join(f"{sample_out_dir}", f"out-{file_idx}.npz")
                numpy.savez(sample_output_filename, *sample_out)
                exported_samples += 1

                if exported_samples >= num_export_samples:
                    break

            if exported_samples >= num_export_samples:
                break

        _LOGGER.info(
            f"Exported {exported_samples} samples to {save_dir}"
        )
