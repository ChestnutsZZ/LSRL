"""
离散分类特征类
"""
from pandas import Series
from pandas.api import types
from torch import Tensor
from typing import Dict, Any, Optional

from torchrec.feature_column.CategoricalColumn import CategoricalColumn
from torchrec.utils.const import *


class CategoricalColumnWithIdentity(CategoricalColumn):
    """离散分类特征类，输入必须是整数，输入不建议过于稀疏，等价于生成[0,category_num)的one-hot向量"""

    def __init__(self, category_num: int, feature_name: str):
        super().__init__(category_num)
        self.feature_name = feature_name

    def get_feature_data(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        """获取特征数据"""
        return batch.get(self.feature_name).long()

    @staticmethod
    def from_series(feature_name: str, series: Series, other_info: Optional[Dict[str, Any]] = None):
        """从pandas.Series中构造"""
        assert types.is_integer_dtype(series), series.dtypes
        column = CategoricalColumnWithIdentity(
            feature_name=feature_name,
            category_num=series.max() + 1
        )
        column.set_info(MIN, series.min())
        column.set_info(MAX, series.max())
        if other_info:
            for key, value in other_info.items():
                column.set_info(key, value)
        return column

    def __str__(self):
        s = f"name: {self.feature_name}, category_num: {self.category_num}"
        for key, value in self.get_info().items():
            s += f", {key}: {value}"
        return s

    def __repr__(self):
        return str(self)
