from typing import List

from torchrec.metric import IMetric


class MetricList:
    def __init__(self, metrics: List[IMetric]):
        self.user_sample_n: int = metrics[0].user_sample_n
        for metric in metrics:
            assert self.user_sample_n == metric.user_sample_n
        self.metrics: List[IMetric] = metrics

    def __call__(self, prediction, target):
        pos_rank = self.metrics[0].get_pos_rank(prediction)
        return {metric.name: metric.fast_calc(pos_rank) for metric in self.metrics}
