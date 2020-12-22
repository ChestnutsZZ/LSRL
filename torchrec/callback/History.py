"""
训练历史
"""
from typing import List, Optional, Dict

import numpy as np

from torchrec.callback.ICallback import ICallback


class History(ICallback):
    """
    训练历史记录
    该回调会自动应用于IRecommender.fit函数，并返回
    """

    def __init__(self):
        super().__init__()
        self.epoch: List[int] = []
        self.history = {}

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始的时候"""
        self.epoch: List[int] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束时调用"""
        logs: Dict = logs or {}
        self.epoch.append(epoch + 1)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.model.history = self

    def get_best_epoch_logs(self, monitor: str, monitor_mode: str = "min"):
        """获取监视指标最好的epoch与指标"""
        monitor_value_array = np.array(self.history[monitor])
        if monitor_mode == "max":
            monitor_value_array = -monitor_value_array
        best_index = np.argsort(monitor_value_array)[0]
        return self.epoch[best_index], {key: self.history[key][best_index] for key in self.history}
