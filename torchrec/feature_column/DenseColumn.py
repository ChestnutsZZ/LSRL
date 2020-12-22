"""
密集特征列，能够输入深度网络
"""
from abc import ABC

from torchrec.feature_column.FeatureColumn import FeatureColumn


class DenseColumn(FeatureColumn, ABC):
    """密集特征列"""
    pass
