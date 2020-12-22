"""
离散分类特征列，只能输入线性网络，输入深度网络需要通过Embedding等操作，除非模型有隐式转换
"""
from abc import ABC

from torchrec.feature_column.FeatureColumn import FeatureColumn


class CategoricalColumn(FeatureColumn, ABC):
    """离散分类特征类"""

    def __init__(self, category_num: int):
        super().__init__()
        self.category_num = category_num
