#!/usr/bin/env python

class Callbacks:
    """"
    Handles all registered callbacks for YOLOv5 Hooks
    """

    _callbacks = {
        'on_pretrain_routine_start' :[],
        'on_pretrain_routine_end' :[],

        'on_train_start' :[],
        'on_train_end': [],
        'on_train_epoch_start': [],
        'on_train_epoch_end': [],
        'on_train_batch_start': [],
        'on_train_batch_end': [],
        
        'on_val_start' :[],
        'on_val_end': [],
        'on_val_epoch_start': [],
        'on_val_epoch_end': [],
        'on_val_batch_start': [],
        'on_val_batch_end': [],


        'on_model_save': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'teardown': [],
    }

    def __init__(self):
        return

    def regsiterAction(self, hook, name, callback):
        """
        Register a new action to a callback hook

        Args:
            action      The callback hook name to register the action to
            name        The name of the action
            callback    The callback to fire

        Returns:
            (Bool)      The success state     
        """
        if hook in self._callbacks:
            self._callbacks[hook].append({'name': name, 'callback': callback})
            return True
        else:
            return False

    def getRegisteredActions(self, hook=None):
        """"
        Returns all the registered actions by callback hook

        Args:
            hook The name of the hook to check, defaults to all
        """
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def fireCallbacks(self, register, *args):
        """
        Loop throughs the registered actions and fires all callbacks
        """
        for logger in register:
            logger['callback'](*args)


    def on_pretrain_routine_start(self, *args):
        """
        Fires all registered callbacks at the start of each pretraining routine
        """
        self.fireCallbacks(self._callbacks['on_pretrain_routine_start'], *args)

    def on_pretrain_routine_end(self, *args):
        """
        Fires all registered callbacks at the end of each pretraining routine
        """
        self.fireCallbacks(self._callbacks['on_pretrain_routine_end'], *args)

    def on_train_start(self, *args):
        """
        Fires all registered callbacks at the start of each training
        """
        self.fireCallbacks(self._callbacks['on_train_start'], *args)

    def on_train_end(self, *args):
        """
        Fires all registered callbacks at the end of training
        """
        self.fireCallbacks(self._callbacks['on_train_end'], *args)

    def on_train_epoch_start(self, *args):
        """
        Fires all registered callbacks at the start of each training epoch
        """
        self.fireCallbacks(self._callbacks['on_train_epoch_start'], *args)

    def on_train_epoch_end(self, *args):
        """
        Fires all registered callbacks at the end of each training epoch
        """
        self.fireCallbacks(self._callbacks['on_train_epoch_end'], *args)


    def on_train_batch_start(self, *args):
        """
        Fires all registered callbacks at the start of each training batch
        """
        self.fireCallbacks(self._callbacks['on_train_batch_start'], *args)

    def on_train_batch_end(self, *args):
        """
        Fires all registered callbacks at the end of each training batch
        """
        self.fireCallbacks(self._callbacks['on_train_batch_end'], *args)

    def on_val_start(self, *args):
        """
        Fires all registered callbacks at the start of the validation
        """
        self.fireCallbacks(self._callbacks['on_val_start'], *args)

    def on_val_end(self, *args):
        """
        Fires all registered callbacks at the end of the validation
        """
        self.fireCallbacks(self._callbacks['on_val_end'], *args)

    def on_val_epoch_start(self, *args):
        """
        Fires all registered callbacks at the start of each validation epoch
        """
        self.fireCallbacks(self._callbacks['on_val_epoch_start'], *args)

    def on_val_epoch_end(self, *args):
        """
        Fires all registered callbacks at the end of each validation epoch
        """
        self.fireCallbacks(self._callbacks['on_val_epoch_end'], *args)

    def on_val_batch_start(self, *args):
        """
        Fires all registered callbacks at the start of each validation batch
        """
        self.fireCallbacks(self._callbacks['on_val_batch_start'], *args)

    def on_val_batch_end(self, *args):
        """
        Fires all registered callbacks at the end of each validation batch
        """
        self.fireCallbacks(self._callbacks['on_val_batch_end'], *args)

    def on_model_save(self, *args):
        """
        Fires all registered callbacks after each model save
        """
        self.fireCallbacks(self._callbacks['on_model_save'], *args)

    def optimizer_step(self, *args):
        """
        Fires all registered callbacks on each optimizer step
        """
        self.fireCallbacks(self._callbacks['optimizer_step'], *args)

    def on_before_zero_grad(self, *args):
        """
        Fires all registered callbacks before zero grad
        """
        self.fireCallbacks(self._callbacks['on_before_zero_grad'], *args)

    def teardown(self, *args):
        """
        Fires all registered callbacks before teardown
        """
        self.fireCallbacks(self._callbacks['teardown'], *args)

    
