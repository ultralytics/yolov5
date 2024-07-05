# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Callback utils."""

import threading


class Callbacks:
    """Handles all registered callbacks for YOLOv5 Hooks."""

    def __init__(self):
        """
        Initializes a Callbacks object to manage registered YOLOv5 training event hooks.

        Initializes a dictionary of callback lists, each corresponding to specific training and validation events,
        allowing the registration and invocation of custom functions at various points in the training pipeline.
        The supported callback keys include events such as `on_train_start`, `on_train_epoch_end`, and `on_val_batch_start`.

        Attributes:
            _callbacks (dict[str, list[callable]]): A dictionary with event names as keys and lists of callback functions
                                                    as values. Each list contains functions to be executed when the
                                                    corresponding event is triggered.
        """
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
        Registers a new action to a specific callback hook.

        Args:
            hook (str): The name of the callback hook to which the action will be registered. Must be one of
                the predefined hooks in the Callbacks class.
            name (str, optional): An optional name for the action to allow for easier reference. Leave empty
                if no specific name is required.
            callback (Callable): The callback function to be executed when the specified hook is triggered.
                Must be a callable object.

        Returns:
            None

        Raises:
            AssertionError: If the specified `hook` does not exist in the predefined callback hooks.
            AssertionError: If the provided `callback` is not callable.

        Notes:
            - The available hooks include 'on_pretrain_routine_start', 'on_pretrain_routine_end', 'on_train_start',
              'on_train_epoch_start', 'on_train_batch_start', 'optimizer_step', 'on_before_zero_grad',
              'on_train_batch_end', 'on_train_epoch_end', 'on_val_start', 'on_val_batch_start', 'on_val_image_end',
              'on_val_batch_end', 'on_val_end', 'on_fit_epoch_end', 'on_model_save', 'on_train_end',
              'on_params_update', and 'teardown'.

        Examples:
            ```python
            def my_custom_callback(trainer):
                print("Custom callback executed")

            callbacks = Callbacks()
            callbacks.register_action('on_train_start', name='my_callback', callback=my_custom_callback)
            ```
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """
        Returns all the registered actions for a given callback hook or all hooks if no hook is specified.

        Args:
            hook (str | None): The name of the callback hook to retrieve registered actions for. If None, retrieves actions for
                all hooks.

        Returns:
            dict | list: A dictionary containing lists of registered actions indexed by hook names if `hook` is None.
                Otherwise, a list of registered actions for the specified hook.

        Examples:
        ```python
        callbacks = Callbacks()
        actions = callbacks.get_registered_actions('on_train_start')
        # returns a list of registered actions for the 'on_train_start' hook
        ```
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        Loop through the registered actions and fire all callbacks on the main thread or in a separate thread.

        Args:
            hook (str): The name of the callback hook to trigger.
            args (tuple): Positional arguments to be passed to the callback functions.
            thread (bool): If True, run callbacks in a separate daemon thread. Defaults to False.
            kwargs (dict): Keyword arguments to be passed to the callback functions.

        Returns:
            None

        Raises:
            AssertionError: If the given hook is not found in the registered callbacks.

        Example:
            ```python
            # Assuming 'callbacks' is an instance of the Callbacks class
            callbacks.run('on_train_start', arg1, arg2, thread=True, kwarg1=val1, kwarg2=val2)
            ```

        Notes:
            - Callbacks are executed in the order they were registered.
            - When `thread` is set to True, callbacks are executed in separate daemon threads.

        Links:
            For more details, refer to the Ultralytics YOLOv5 repository on GitHub:
            https://github.com/ultralytics/ultralytics
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
