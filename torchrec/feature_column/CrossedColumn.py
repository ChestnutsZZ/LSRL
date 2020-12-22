"""
分类特征交叉列
"""
from typing import Dict, Any, List

from torch import Tensor

from torchrec.feature_column.CategoricalColumn import CategoricalColumn


class CrossedColumn(CategoricalColumn):
    """分类特征交叉列，输入必须是已经创建好的分类特征列"""

    def __init__(self, categorical_columns: List[CategoricalColumn]):
        category_num = 1
        for categorical_column in categorical_columns:
            category_num *= categorical_column.category_num
        super().__init__(category_num)
        self.categorical_columns = categorical_columns
        self.coefficients = [1] * len(categorical_columns)
        for i in range(len(categorical_columns) - 1, 0, -1):
            self.coefficients[i - 1] = self.coefficients[i] * self.categorical_columns[i].category_num

    def get_feature_data(self, batch: Dict[str, Any]) -> Tensor:
        """获取特征数据"""
        return sum((self.coefficients[i] * self.categorical_columns[i].get_feature_data(batch)
                    for i in range(len(self.categorical_columns))))
