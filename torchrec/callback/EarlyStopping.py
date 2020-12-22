"""
提前停止
"""
from typing import Optional, Dict

import numpy as np

from torchrec.callback.ICallback import ICallback


class EarlyStopping(ICallback):
    """提前停止"""

    def __init__(self,
                 monitor: str = 'val_loss',
                 min_delta: float = 0.0,
                 patience: int = 0,
                 verbose: int = 0,
                 mode: str = 'min',
                 baseline: float = None):
        """
        :param monitor: 监视的指标
        :param min_delta: 最小更新值，小于该值不视为提升
        :param patience: 最多容忍几个epoch不提升
        :param verbose: 0 / 1
        :param mode: "min" / "max"
        :param baseline: 基准
        """
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f'mode参数值不合法: {mode}')

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_begin(self, logs: Optional[Dict] = None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            raise ValueError(f'monitor参数值不在记录范围中: {self.monitor}')
        return monitor_value
