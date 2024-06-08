# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Callback utils."""

import threading


class Callbacks:
    """Handles all registered callbacks for YOLOv5 Hooks."""

    def __init__(self):
        """Initializes a Callbacks object to manage registered YOLOv5 training event hooks."""
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  # fit = train + val
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook.

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """
        Returns all the registered actions by callback hook.

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        Loop through the registered actions and fire all callbacks on main thread.

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            thread: (boolean) Run callbacks in daemon thread
            kwargs: Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
