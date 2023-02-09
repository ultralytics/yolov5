import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

from utils.autobatch import check_train_batch_size
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.neuralmagic.quantization import update_model_bottlenecks
from utils.neuralmagic.utils import ALMOST_ONE, ToggleableModelEMA, load_ema, nm_log_console
from utils.torch_utils import ModelEMA, de_parallel

__all__ = [
    "SparsificationManager",
    "maybe_create_sparsification_manager",
    "apply_recipe_one_shot",
]


class SparsificationManager(object):
    """
    Class for managing train state during sparse training with Neural Magic

    :param model: model to be trained
    :param train_recipe: yaml string or path to recipe to apply during training
    :param recipe_args: additional arguments to override any root variables
        in the recipe with (i.e. num_epochs, init_lr)
    :param checkpoint_recipe: yaml string or path to recipe previously used to create
        loaded model, if any
    :param last_epoch: last training epoch run for loaded model, relative to checkpoint
        recipe
    :param device: device to load model to
    :param resumed: True for runs continued with the --resume flag
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_recipe: Optional[str],
        recipe_args: Optional[Union[Dict[str, Any], str]],
        checkpoint_recipe: Optional[str] = None,
        last_epoch: int = 0,
        device: Union[str, torch.device] = "cpu",
        resumed: bool = False,
    ):
        self.loggers = None
        self.compute_loss = None
        self.qat_started = False
        self.current_phase = None
        self.passed_phases = []

        # Recipes can be sensitive to module names, target correct submodule if parallel
        self.model = (
            model.module
            if (
                type(model)
                in (
                    torch.nn.parallel.DataParallel,
                    torch.nn.parallel.DistributedDataParallel,
                )
            )
            else model
        )

        # update bottleneck modules to have quantizable add nodes
        model = update_model_bottlenecks(model).to(device)

        # Training manager created from checkpoint recipe, if any
        self.checkpoint_manager = (
            ScheduledModifierManager.from_yaml(checkpoint_recipe)
            if checkpoint_recipe
            else None
        )

        # Training manager for current training run
        self.train_manager = (
            ScheduledModifierManager.from_yaml(
                file_path=train_recipe, recipe_variables=recipe_args
            )
            if train_recipe
            else None
        )

        # Apply recipe structure from checkpoint. Can include QAT and layer thinning
        if self.checkpoint_manager:
            self.checkpoint_manager.apply_structure(
                self.model, last_epoch + ALMOST_ONE if last_epoch >= 0 else float("inf")
            )

        # Process sparsification settings and verify that they are a valid combination
        self.set_sparsification_info()
        if not resumed:
            self.check_for_invalid_state()

    def set_sparsification_info(self):
        """
        Set attributes relating to sparsification in run
        """
        self.has_pruning_phase = bool(
            self.train_manager and self.train_manager.pruning_modifiers
        )
        self.has_qat_phase = bool(
            self.train_manager and self.train_manager.quantization_modifiers
        )
        self.pruned_checkpoint = bool(
            self.checkpoint_manager and self.checkpoint_manager.pruning_modifiers
        )
        self.quantized_checkpoint = bool(
            self.checkpoint_manager and self.checkpoint_manager.quantization_modifiers
        )
        self.has_distillation_phase = bool(
            self.train_manager and self.train_manager.distillation_modifiers
        )

        self.first_pruning_epoch = (
            math.floor(
                min([mod.start_epoch for mod in self.train_manager.pruning_modifiers])
            )
            if self.has_pruning_phase
            else None
        )
        self.last_pruning_epoch = (
            math.floor(
                min([mod.end_epoch for mod in self.train_manager.pruning_modifiers])
            )
            if self.has_pruning_phase
            else None
        )
        self.first_qat_epoch = (
            math.floor(
                min(
                    [
                        mod.start_epoch
                        for mod in self.train_manager.quantization_modifiers
                    ]
                )
            )
            if self.has_qat_phase
            else None
        )
        self.first_distillation_epoch = (
            math.floor(
                min(
                    [
                        mod.start_epoch
                        for mod in self.train_manager.distillation_modifiers
                    ]
                )
            )
            if self.has_distillation_phase
            else None
        )

    def check_for_invalid_state(self):
        """
        Checks that the training sparsification recipe (or lack of) is a valid recipe
        for the loaded model. This primarily applies when the loaded model is already
        sparsified.
        """

        # Checking valid state for pruned models
        if self.pruned_checkpoint:
            if not self.train_manager:
                self.log_console(
                    "Pruned model was loaded, but no sparsification recipe detected - "
                    "model may revert to dense state. A recipe with a "
                    "ConstantPruningModifier can be used to maintain model sparsity "
                    "while training",
                    level="warning",
                )
            elif not self.has_pruning_phase:
                self.log_console(
                    "Pruned model was loaded, but no pruning modifiers detected in "
                    "sparsification recipe - model may revert to dense state. A "
                    "recipe with a ConstantPruningModifier can be used to maintain "
                    "model sparsity while training",
                    level="warning",
                )

        # Checking valid state for quantized models
        if self.quantized_checkpoint and self.has_qat_phase:
            raise ValueError(
                "Quantization can not be applied more than once. Loaded quantized "
                "model from checkpoint and detected quantization modifier in "
                "sparsification recipe. This is unsupported behavior. Ending run."
            )

    def initialize(
        self,
        loggers: Optional[Loggers],
        scaler: torch.cuda.amp.GradScaler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        ema: Optional[ModelEMA],
        start_epoch: int,
        steps_per_epoch: int,
        epochs: int,
        compute_loss: ComputeLoss,
        distillation_teacher: Optional[torch.nn.Module],
        ema_kwargs: Dict[str, Any] = {},
        resumed: bool = False,
    ) -> Tuple[
        torch.cuda.amp.GradScaler, torch.optim.lr_scheduler._LRScheduler, ModelEMA, int
    ]:
        """
        Update objects controlling the training process for sparse training
        """
        # Wrap model for sparse training modifiers from recipe
        if self.train_manager:
            self.train_manager.initialize(
                module=self.model,
                epoch=start_epoch,
                distillation_teacher=distillation_teacher,
            )
            self.steps_per_epoch = steps_per_epoch

            # initialize SparseML loggers, including recipe modifier loggers
            if loggers:
                self.initialize_loggers(loggers)

            self.log_console(
                "Sparse training detected. Wrapping training process with SparseML"
            )

            # If resumed run, apply recipe structure up to last epoch run. Structure can
            # include QAT and layer thinning
            if resumed:
                self.train_manager.apply_structure(
                    self.model, start_epoch - 1 + ALMOST_ONE
                )

            # Wrap the scaler for sparse training modifiers from recipe
            scaler = self.train_manager.modify(
                self.model,
                optimizer,
                steps_per_epoch=self.steps_per_epoch,
                wrap_optim=scaler,
            )

            # If recipe contains lr modifiers, turn off native lr scheduler
            if self.train_manager.learning_rate_modifiers:
                scheduler = None
                self.log_console(
                    "Disabling LR scheduler, managing LR using SparseML recipe"
                )

            # If recipe contains epoch range modifiers, overwrite epoch range
            if self.train_manager.epoch_modifiers and self.train_manager.max_epochs:
                epochs = self.train_manager.max_epochs
                self.log_console(
                    "Overriding total number of training epochs with value from "
                    f"recipe: {epochs}"
                )

        # construct a ToggleableModelEMA from ModelEMA, allowing for on/off toggle
        if ema:
            # QAT is active at the start epoch, disable ema
            qat_active = (
                self.has_qat_phase and start_epoch >= self.first_qat_epoch
            )

            ema = load_ema(
                ema.ema.state_dict(),
                self.model if not qat_active else ema.ema,
                **ema_kwargs,
            )

        self.optimizer = optimizer
        self.compute_loss = compute_loss

        return scaler, scheduler, ema, epochs

    def initialize_loggers(self, loggers: Loggers):
        """
        Initialize SparseML console, wandb, and tensorboard loggers from YOLOv5 loggers
        """
        # Console logger
        self.loggers = loggers.logger

        # For logging sparse training values (e.g. sparsity %, custom lr schedule, etc.)
        def _logging_lambda(tag, value, values, step, wall_time, level):
            if not loggers.wandb or not loggers.wandb.wandb:
                return

            if value is not None:
                loggers.wandb.log({tag: value})

            if values:
                loggers.wandb.log(values)

        self.train_manager.initialize_loggers(
            [
                SparsificationGroupLogger(
                    lambda_func=_logging_lambda,
                    tensorboard=loggers.tb,
                )
            ]
        )

        # Attach recipe to wandb log
        if loggers.wandb and loggers.wandb.wandb:
            artifact = loggers.wandb.wandb.Artifact("recipe", type="recipe")
            with artifact.new_file("recipe.yaml") as file:
                file.write(str(self.train_manager))
            loggers.wandb.wandb.log_artifact(artifact)

    def get_final_checkpoint_recipe(self) -> ScheduledModifierManager:
        """
        Return the final ScheduledModifierManager that would be saved with final
        models. Represents all recipes applied to model, allowing for multiple stages
        of sparse training
        """
        return (
            ScheduledModifierManager.compose_staged(
                self.checkpoint_manager, self.train_manager
            )
            if self.checkpoint_manager and self.train_manager
            else self.train_manager or self.checkpoint_manager
        )

    def starting_qat(self, epoch: float) -> bool:
        """
        Returns true if this is the first epoch QAT is turned on
        """
        # Continued training of quantized model
        if self.quantized_checkpoint:
            return True

        # Training with a quantization recipe
        if not self.qat_started:
            self.qat_started = self.qat_active(epoch)
            return self.qat_started
        else:
            return False

    def qat_active(self, epoch: float) -> bool:
        """
        Returns true if QAT is turned on for the given epoch

        :param epoch: epoch to check QAT status for
        """
        if self.has_qat_phase:
            return self.first_qat_epoch < epoch + 1
        else:
            return False

    def distillation_active(self, epoch: float) -> bool:
        """
        Returns true if distillation is turned on for the given epoch

        :param epoch: epoch to check distillation status for
        """
        if self.has_distillation_phase:
            return self.first_distillation_epoch < epoch + 1
        else:
            return False

    def log_console(self, message: str, level: str = "info"):
        """
        Pass through to nm_log_console with manager logger, if initialized

        :param message: message to be logged
        :param level: level to be logged at
        """
        nm_log_console(message=message, logger=self.loggers, level=level)

    def disable_ema_amp(
        self,
        ema: Optional[ToggleableModelEMA],
        amp: bool,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Tuple[ToggleableModelEMA, bool, torch.cuda.amp.GradScaler]:
        """
        Disable EMA and AMP if active, as they're not compatible with QAT
        """
        self.log_console("Starting QAT phase")
        if ema and ema.enabled:
            self.log_console("Turning off EMA (not supported with QAT)")
            ema.enabled = False
        if amp:
            self.log_console("Turning off AMP (not supported with QAT)")
            amp = False
            scaler._enabled = False

        return ema, amp, scaler

    def rescale_gradient_accumulation(
        self, batch_size: int, accumulate: int, image_size: int
    ) -> Tuple[int, int]:
        """
        Used when autobatch and QAT are both enabled. Training with QAT adds additional
        overhead which can cause OOM errors if autobatch is enabled. This function
        rescales batch size and gradient accumulation to fit into memory with QAT while
        maintaining the original effective batch size
        """
        # Temporary copy of the model with QAT applied
        quant_model_copy = deepcopy(de_parallel(self.model))
        train_manager_copy = ScheduledModifierManager.from_yaml(str(self.train_manager))
        train_manager_copy.apply_structure(quant_model_copy, float("inf"))

        # batch size to maintain
        effective_batch_size = batch_size * accumulate

        # Calculate maximum batch size that will fit in memory
        new_batch_size = check_train_batch_size(quant_model_copy, image_size, False)

        # Roughly calculate batch size by rounding. In many circumstances this can
        # result in an effective batch size that is 1-few off from the original
        new_accumulate = max(round(effective_batch_size / new_batch_size), 1)
        new_batch_size = max(round(effective_batch_size / new_accumulate), 1)

        self.log_console(
            f"Batch size rescaled to {new_batch_size} with {new_accumulate} gradient "
            "accumulation steps for QAT"
        )

        if new_accumulate * new_batch_size != batch_size * accumulate:
            self.log_console(
                "New effective batch size doesn't match previous effective batch size. "
                f"Previous effective batch size: {batch_size * accumulate}. "
                f"New effective batch size: {new_batch_size * new_accumulate}",
                level="warning",
            )

        return new_batch_size, new_accumulate

    def compute_distillation_loss(
        self, epoch: int, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distillation loss

        :param epoch: current epoch
        :param inputs: inputs to the student model
        :param targets: input labels

        :return: computed distillation loss and loss items
        """
        batch_size = inputs.size(0)

        # Compute student-only loss
        student_outputs = self.model(inputs)
        loss, loss_items = self.compute_loss(student_outputs, targets)

        # Compute full distillation loss
        loss = loss / batch_size
        loss = self.train_manager.loss_update(
            loss=loss,
            module=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            steps_per_epoch=self.steps_per_epoch,
            student_outputs=student_outputs,
            student_inputs=inputs,
            student_labels=targets,
        )
        loss = loss * batch_size

        return loss, loss_items

    def update_state_dict_for_saving(
        self,
        ckpt: Dict[str, Any],
        final_epoch: bool,
        ema_enabled: bool,
        number_classes: int,
    ) -> Dict[str, Any]:
        """
        Update checkpoint dictionary to be compatible with sparse model saving

        :param ckpt: original checkpoint dictionary
        :param final_epoch: True if called after last training epoch
        :param ema_enabled: True if ema is turned on
        :param number_classes: Number of classes model detects
        """
        # checkpoint recipe saved with final models, for state re-construction upon
        # loading for validation or additional stage of sparsification
        checkpoint_recipe = self.get_final_checkpoint_recipe() if final_epoch else None

        # Pickling is not supported for quantized models for a subset of the supported
        # torch versions, thus all sparse models are saved via their state dict
        sparseml_dict_update = {
            "model": ckpt["model"].state_dict(),
            "yaml": ckpt["model"].yaml,
            "ema": ckpt["ema"].state_dict() if ema_enabled else None,
            "updates": ckpt["updates"] if ema_enabled else None,
            "checkpoint_recipe": str(checkpoint_recipe) if checkpoint_recipe else None,
            "epoch": -1 if final_epoch else ckpt["epoch"],
            "nc": number_classes,
        }
        ckpt.update(sparseml_dict_update)

        return ckpt

    def maybe_switch_phases(
        self, epoch: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Check if a new phase has been entered. If it has, record the new phase and
        reset the tracked best fitness. Possible phases are dense, pruned,
        pruned_quantized, and quantized

        :param epoch: current epoch
        :return: if new phase entered, then new best_fitness of 0.0 and filename to save
            best models to is returned. Otherwise, None returned for each
        """
        # QAT + Pruning run
        if self.has_pruning_phase and self.has_qat_phase:
            if epoch < self.first_pruning_epoch:
                current_phase = "dense"
            elif self.first_pruning_epoch <= epoch < self.first_qat_epoch:
                current_phase = "pruned"
            else:
                current_phase = "pruned_quantized"

        # Pruning only run
        elif self.has_pruning_phase:
            current_phase = "pruned" if epoch >= self.first_pruning_epoch else "dense"

        # QAT only run
        elif self.has_qat_phase:
            current_phase = "quantized" if epoch >= self.first_qat_epoch else "dense"

        # Run with no sparsification modifiers
        else:
            current_phase = "dense"

        # Update phase
        if self.current_phase != current_phase:
            self.passed_phases.append(current_phase)
            self.current_phase = current_phase

            return 0.0, f"best_{self.current_phase}.pt"

        return None, None

    def strip_sparsified_optimizer(
        self, checkpoint_path: str = "best.pt", save_name: Optional[str] = None
    ):
        """
        Updates the saved state dict to reflect a final saved state. Fulfills the same
        function as utils.general.strip_optimizer() for sparsified checkpoints

        :param checkpoint_path: path to the checkpoint to be loaded
        :param save_name: optional path to save the "stripped" checkpoint to
        """
        ckpt = torch.load(checkpoint_path)

        if ckpt.get("ema"):
            ckpt["model"] = ckpt["ema"]
        for key in "optimizer", "best_fitness", "ema", "updates":
            if key in ckpt:
                ckpt[key] = None

        ckpt["checkpoint_recipe"] = str(self.get_final_checkpoint_recipe())

        torch.save(ckpt, save_name or checkpoint_path)

        megabytes = os.path.getsize(save_name or checkpoint_path) / 1e6
        self.log_console(
            f"Optimizer stripped from {checkpoint_path},"
            f"{f' saved as {save_name},' if save_name else ''} {megabytes:.1f}MB"
        )


def maybe_create_sparsification_manager(
    model: torch.nn.Module,
    ckpt: Dict[str, Any],
    train_recipe: Optional[str],
    recipe_args: Optional[Union[Dict[str, Any], str]],
    device: Union[str, torch.device],
    resumed: bool = False,
) -> Optional[SparsificationManager]:
    """
    If sparse training or checkpoint detected, load sparse model and return
    SparsificationManager object. Otherwise do nothing.

    :param model: skeleton model
    :param ckpt: loaded checkpoint
    :param train_recipe: yaml string or path to recipe to apply during training
    :param recipe_args: additional arguments to override any root variables
        in the recipe with (i.e. num_epochs, init_lr)
    :param device: device to load model to
    :param resumed: True for runs continued with the --resume flag
    """
    if "recipe" in ckpt:
        ckpt = _make_legacy_checkpoint_compatible(ckpt)

    if ckpt.get("checkpoint_recipe") or train_recipe:

        sparsification_manager = SparsificationManager(
            model=model,
            train_recipe=train_recipe,
            recipe_args=recipe_args,
            checkpoint_recipe=ckpt.get("checkpoint_recipe"),
            last_epoch=ckpt["epoch"],
            device=device,
            resumed=resumed,
        )

        # reconstruct ToggleableModelEMA from state dictionary
        if ckpt["ema"]:
            ckpt["ema"] = load_ema(ckpt["ema"], model)

        return sparsification_manager

    else:
        return None


def apply_recipe_one_shot(model: torch.nn.Module, recipe: str) -> SparsificationManager:
    """
    Applies a recipe to a model in one-shot, applying any pruning and quantization
    modifiers.

    NOTE: the current implementation is zero-shot. One-shot application with data to
        be added in the near future
    """
    model.train()
    sparsification_manager = SparsificationManager(
        model=model,
        train_recipe=None,
        recipe_args=None,
        checkpoint_recipe=recipe,
        last_epoch=-1,
    )
    model.eval()

    return sparsification_manager


def _make_legacy_checkpoint_compatible(checkpoint: Dict[str, Any]):
    """
    Update a legacy sparsezoo checkpoint to work with the updated version of yolov5
    """
    checkpoint["checkpoint_recipe"] = checkpoint.pop("recipe")
    return checkpoint
