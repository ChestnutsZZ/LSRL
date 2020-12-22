"""
模型检查点
"""
import numpy as np
from typing import Union, Optional, Dict

from torchrec.callback.ICallback import ICallback


class ModelCheckpoint(ICallback):
    """模型状态检查点"""

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 verbose: int = 0,
                 save_best_only: bool = False,
                 mode: str = 'min',
                 save_freq: Union[str, int] = 'epoch'):
        """
        :param filepath: 保存文件名，可以包含epoch与logs的key在内占位符
        :param monitor: 监视哪一个指标
        :param verbose: 0/1
        :param save_best_only: 是否只在指标变得更好时保存模型状态
        :param mode: 指定指标变好的方向，"min"或者"max"
        :param save_freq: 保存频率，如果为"epoch"，则每个epoch末检查；如果是整数，则每固定数量batch检查
        """
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self._current_epoch = 0
        self.epochs_since_last_saving = 0
        self._batches_seen_since_last_saving = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f'mode参数值不合法: {mode}')

        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
            raise ValueError(f'save_freq参数值不合法: {self.save_freq}')

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if isinstance(self.save_freq, int):
            logs = logs or {}
            self._batches_seen_since_last_saving += 1
            if self._batches_seen_since_last_saving >= self.save_freq:
                self._save_model(epoch=self._current_epoch, logs=logs)
                self._batches_seen_since_last_saving = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        self.epochs_since_last_saving += 1
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, logs=logs)

    def _save_model(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_saving >= 1:
            self.epochs_since_last_saving = 0
            filepath = self._get_file_path(epoch, logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    raise ValueError(f'monitor参数值不在记录范围中: {self.monitor}')
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch + 1:05d}: {self.monitor} improved from {self.best:0.5f}'
                                  f' to {current:0.5f}, saving model to {filepath}')
                        self.best = current
                        self.model.save_weights(filepath)
                        self.model.save_best_weights()
                    else:
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch + 1:05d}: {self.monitor} did not improve from {self.best:0.5f}')
            else:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1:05d}: saving model to {filepath}')
                self.model.save_weights(filepath)
                self.model.save_best_weights()

    def _get_file_path(self, epoch, logs):
        try:
            return self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
