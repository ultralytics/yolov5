import logging
import math
import random
import os
from typing import Any, Dict, Optional
from re import search
from sparsezoo import Model
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger
from sparseml.pytorch.sparsification.quantization import QuantizationModifier
import torchvision.transforms.functional as F

from utils.torch_utils import is_parallel
import torch.nn as nn
import torch
import numpy

_LOGGER = logging.getLogger(__file__)

def _get_model_framework_file(model, path):
    available_files = model.training.default.files
    transfer_request = search("recipe(.*)transfer", path)
    checkpoint_available = any([".ckpt" in file.name for file in available_files])
    final_available = any([not ".ckpt" in file.name for file in available_files])

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if available and requested
        return [file for file in available_files if ".ckpt" in file.name][0]
    elif final_available:
        # default to returning final state, if available
        return [file for file in available_files if ".ckpt" not in file.name][0]

    raise ValueError(f"Could not find a valid framework file for {path}")


def check_download_sparsezoo_weights(path):
    if isinstance(path, str):
        if path.startswith("zoo:"):
            # load model from the SparseZoo and override the path with the new download
            model = Model(path)
            file = _get_model_framework_file(model, path)
            path = file.path

        return path

    if isinstance(path, list):
        return [check_download_sparsezoo_weights(p) for p in path]

    return path


class SparseMLWrapper(object):
    def __init__(
        self, 
        model, 
        checkpoint_recipe, 
        train_recipe,
        recipe_args = None,
        train_mode=False,
        epoch=-1, 
        steps_per_epoch=-1, 
        one_shot=False,
        ):
        self.enabled = bool(train_recipe)
        self.model = model.module if is_parallel(model) else model
        self.checkpoint_manager = ScheduledModifierManager.from_yaml(checkpoint_recipe) if checkpoint_recipe else None
        self.manager = ScheduledModifierManager.from_yaml(train_recipe, recipe_variables=recipe_args) if train_recipe else None
        self.logger = None
        self.start_epoch = None
        self.steps_per_epoch = steps_per_epoch
        self.one_shot = one_shot
        self.train_recipe = train_recipe

        self.apply_checkpoint_structure(train_mode, epoch, one_shot)

    def state_dict(self, final_epoch):
        if self.enabled and final_epoch:
            checkpoint_recipe = (
                str(ScheduledModifierManager.compose_staged(self.checkpoint_manager, self.manager)) 
                if self.checkpoint_manager else str(self.manager)
            )
            train_recipe = None
        else:
            checkpoint_recipe = str(self.checkpoint_manager) if self.checkpoint_manager else None
            train_recipe = str(self.manager) if self.manager else None
            
        return {
            'checkpoint_recipe': checkpoint_recipe,
            'train_recipe': train_recipe
        }

    def apply_checkpoint_structure(self, train_mode, epoch, one_shot=False):
        if self.checkpoint_manager:
            # if checkpoint recipe has a QAT modifier and this is a transfer learning 
            # run then remove the QAT modifier from the manager 
            if train_mode:
                qat_idx = next((
                    idx for idx, mod in enumerate(self.checkpoint_manager.modifiers) 
                    if isinstance(mod, QuantizationModifier)), -1
                    )
                if qat_idx >= 0:
                    _ = self.checkpoint_manager.modifiers.pop(qat_idx)
        
            self.checkpoint_manager.apply_structure(self.model, math.inf)

        if train_mode and epoch > 0 and self.enabled:
            self.manager.apply_structure(self.model, epoch)
        elif one_shot:
            if self.enabled:
                self.manager.apply(self.model)
                _LOGGER.info(f"Applied recipe {self.train_recipe} in one-shot manner")
            else:
                _LOGGER.info(f"Training recipe for one-shot application not recognized by the manager. Got recipe: "
                            f"{self.train_recipe}"
                            )

    def initialize(
        self, 
        start_epoch, 
        compute_loss, 
        train_loader, 
        device, 
        **train_loader_kwargs
    ):
        if not self.enabled:
            return

        grad_sampler = {
            "data_loader_builder": self._get_data_loader_builder(
                train_loader, device, **train_loader_kwargs
            ),
            "loss_function": lambda pred, target: compute_loss(
                [p for p in pred[1]], target.to(device)
            )[0],
        }

        self.manager.initialize(self.model, start_epoch, grad_sampler=grad_sampler)
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
            epochs = self.manager.max_epochs or epochs  # override num_epochs
            if rank in [0,-1]:
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

    def _get_data_loader_builder(
        self, train_loader, device, multi_scale, img_size, grid_size
    ):
        def _data_loader_builder(kwargs: Optional[Dict[str, Any]] = None):
            template = dict(train_loader.__dict__)

            # drop attributes that will be auto-initialized
            to_drop = [
                k
                for k in template
                if k.startswith("_") or k in ["batch_sampler", "iterator"]
            ]
            for item in to_drop:
                template.pop(item)

            # override defaults if kwargs are given, for example via recipe
            if kwargs:
                template.update(kwargs)
            data_loader = type(train_loader)(**template)

            while True:
                for imgs, targets, *_ in data_loader:
                    imgs = imgs.to(device, non_blocking=True).float() / 255
                    if multi_scale:
                        sz = (
                            random.randrange(img_size * 0.5, img_size * 1.5 + grid_size)
                            // grid_size
                            * grid_size
                        )  # size
                        sf = sz / max(imgs.shape[2:])  # scale factor
                        if sf != 1:
                            ns = [
                                math.ceil(x * sf / grid_size) * grid_size
                                for x in imgs.shape[2:]
                            ]  # new shape (stretched to gs-multiple)
                            imgs = nn.functional.interpolate(
                                imgs, size=ns, mode="bilinear", align_corners=False
                            )
                    yield [imgs], {}, targets
        return _data_loader_builder

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
        image_size: int=640,
        save_inputs_as_uint8: bool = False,
   ):
        save_dir = save_dir or ""
        if not dataloader:
            raise ValueError(
                f"Expected a data loader for exporting samples. Got {dataloader}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        exported_samples = 0

        sample_in_dir = os.path.join(save_dir, "sample_inputs")
        sample_out_dir = os.path.join(save_dir, "sample_outputs")

        os.makedirs(sample_in_dir, exist_ok=True)
        os.makedirs(sample_out_dir, exist_ok=True)
        model_was_in_train_mode = self.model.training
        self.model.eval()

        for _, (images, _, _, _) in enumerate(dataloader):
            images = (
                images.float() / 255
            )  # uint8 to float32, 0-255 to 0.0-1.0
            images = F.resize(images, (image_size, image_size))

            device_imgs = images.to(device, non_blocking=True)
            outs = self.model(device_imgs)

            # Move to cpu for exporting
            ins = images.detach().to("cpu")

            if isinstance(outs, tuple) and len(outs) > 1:
                # flatten into a single list
                outs = [outs[0], *outs[1]]

            outs = [elem.detach().to("cpu") for elem in outs]
            outs_gen = zip(*outs)

            for sample_in, sample_out in zip(ins, outs_gen):
                sample_out = list(sample_out)
                file_idx = f"{exported_samples}".zfill(4)

                sample_input_filename = os.path.join(f"{sample_in_dir}", f"inp-{file_idx}.npz")
                if save_inputs_as_uint8:
                    sample_in = (255 * sample_in).to(dtype=torch.uint8)
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

        # reset model.training if needed
        if model_was_in_train_mode:
            self.model.train()
