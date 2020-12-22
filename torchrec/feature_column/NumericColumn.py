"""
数值列
"""
from typing import Dict, Any

from pandas import Series
from pandas.api import types
from torch import FloatTensor

from torchrec.feature_column.DenseColumn import DenseColumn
from torchrec.feature_column.NormalizationMode import NormalizationMode


class NumericColumn(DenseColumn):
    """数值特征列"""

    def __init__(self, feature_name: str, min_value: float, max_value: float, mean_value: float, std_value: float):
        super().__init__()
        self.feature_name = feature_name
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value

    def get_feature_data(self, batch: Dict[str, Any], normalization_mode: NormalizationMode = NormalizationMode.NOP) \
            -> FloatTensor:
        """获取特征数据"""
        if normalization_mode == NormalizationMode.NOP:
            return batch[self.feature_name].float()
        if normalization_mode == NormalizationMode.MAX_MIN:
            return (batch[self.feature_name].float() - self.min_value) / (self.max_value - self.min_value)
        if normalization_mode == NormalizationMode.Z_SCORE:
            return (batch[self.feature_name].float() - self.mean_value) / self.std_value
        raise Exception("NormalizationMode is wrong!")

    @staticmethod
    def from_series(feature_name: str, series: Series):
        """从pandas.Series中构造"""
        assert types.is_numeric_dtype(series), series.dtypes
        return NumericColumn(
            feature_name=feature_name,
            min_value=series.min(),
            max_value=series.max(),
            mean_value=series.mean(),
            std_value=series.std()
        )

    def __str__(self):
        s = f"name: {self.feature_name}, min: {self.min_value}, max: {self.max_value}, mean: {self.mean_value}" \
            f", std: {self.std_value}"
        for key, value in self.get_info().items():
            s += f", {key}: {value}"
        return s
