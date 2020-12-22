"""
Hit
"""
from numpy import ndarray

from torchrec.metric.IMetric import IMetric


class Hit(IMetric):
    """Hit"""

    def __init__(self, user_sample_n: int, k: int):
        self.k = k
        super().__init__(user_sample_n)
        self.name = f"hit@{self.k}"

    def __call__(self, prediction, target, *args, **kwargs):
        return self.fast_calc(self.get_pos_rank(prediction))

    def fast_calc(self, pos_ranks: ndarray):
        """快速计算"""
        hit_pos_ranks = pos_ranks[pos_ranks <= self.k]
        return len(hit_pos_ranks) / len(pos_ranks)
