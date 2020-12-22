"""
回调列表
"""
import collections
import logging
import time
from typing import List

import numpy as np

from torchrec.callback.History import History
from torchrec.callback.ICallback import ICallback
from torchrec.callback.ProgbarLogger import ProgbarLogger
from torchrec.utils.data_structure import tensor_to_numpy_or_python_type


class ModeKeys:
    TRAIN = 'train'
    TEST = 'test'
    PREDICT = 'predict'


class CallbackList:
    """回调列表"""

    def __init__(self,
                 callbacks: List[ICallback] = None,
                 add_history: bool = False,
                 add_progbar: bool = False,
                 model=None,
                 **params):
        """
        :param callbacks: 回调实例列表
        :param add_history: 是否增加历史回调
        :param add_progbar: 是否增加进度条回调
        :param model: 模型实例
        :param params: 其他参数
        """
        self.callbacks = callbacks or []
        self._add_default_callbacks(add_history, add_progbar)
        self.model = None
        self.params = None

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

        self._queue_length = 10
        self._reset_batch_timing()

        self._should_call_train_batch_hooks = any(
            cb.implements_train_batch_hooks() for cb in self.callbacks)
        self._should_call_test_batch_hooks = any(
            cb.implements_test_batch_hooks() for cb in self.callbacks)
        self._should_call_predict_batch_hooks = any(
            cb.implements_predict_batch_hooks() for cb in self.callbacks)

    def _add_default_callbacks(self, add_history: bool, add_progbar: bool):
        self._progbar = None
        self._history = None

        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger()
            self.callbacks.append(self._progbar)

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

    def _reset_batch_timing(self):
        self._delta_t_batch = 0.
        self._delta_ts = collections.defaultdict(
            lambda: collections.deque([], maxlen=self._queue_length))

    def _process_logs(self, logs):
        """Turns tensors into numpy arrays or Python scalars."""
        if logs:
            return tensor_to_numpy_or_python_type(logs)
        return {}

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for cb in self.callbacks:
            cb.set_params(params)

    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for cb in self.callbacks:
            cb.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
        if hook == 'begin':
            self._t_enter_batch = time.time()
        if hook == 'end':
            # Batch is ending, calculate batch time.
            self._delta_t_batch = time.time() - self._t_enter_batch

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)
        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

        delta_t_median = np.median(self._delta_ts[hook_name])
        if (self._delta_t_batch > 0. and
                delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            logging.warning(
                'Method (%s) is slow compared '
                'to the batch update (%f). Check your callbacks.', hook_name,
                delta_t_median)

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_begin()
        elif mode == ModeKeys.TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_end()
        elif mode == ModeKeys.TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    def on_batch_begin(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._reset_batch_timing()

    def on_epoch_end(self, epoch, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        if self._should_call_test_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        if self._should_call_test_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        if self._should_call_predict_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        if self._should_call_predict_batch_hooks:
            logs = self._process_logs(logs)
            self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)
