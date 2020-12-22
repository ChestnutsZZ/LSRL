"""
评价指标接口类
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray


class IMetric(ABC):
    """评价指标接口类"""

    def __init__(self, user_sample_n: int):
        self.name = "IMetric"
        self.user_sample_n = user_sample_n

    def get_pos_rank(self, prediction) -> ndarray:
        # TopK 验证集与测试集不随机打乱的情况下，每 user_sample_n 个数据是一个用户
        prediction_array: ndarray = prediction.reshape((-1, self.user_sample_n))  # [user_num, user_sample_n]

        # 按照预测值获得正例预测值排序
        # [user_num, user_sample_n]
        sort_idx_array: ndarray = (-prediction_array).argsort()
        # 这里用到一个技巧，每行只有一个正例并且位于第一个，因此只要确认 sort_idx 每行下标 0 的位置即可，然后 +1 确保下标从 1 开始
        pos_ranks: ndarray = np.argwhere(sort_idx_array == 0)[:, 1] + 1  # [user_num]
        return pos_ranks

    @abstractmethod
    def __call__(self, prediction, target, *args, **kwargs):
        pass

    @abstractmethod
    def fast_calc(self, pos_ranks: ndarray):
        """快速计算"""
        pass
