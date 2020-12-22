"""
数据集描述类
"""
from typing import Optional, List, Dict, Union

import numpy as np
from pandas import DataFrame, Series

from torchrec.feature_column import CategoricalColumnWithIdentity, NumericColumn
from torchrec.utils.const import *


class DatasetDescription:
    """数据集描述类"""

    def __init__(self, info: str):
        self.info: str = info

        self.uid_column: Optional[CategoricalColumnWithIdentity] = None
        self.iid_column: Optional[CategoricalColumnWithIdentity] = None
        self.rate_column: Optional[CategoricalColumnWithIdentity] = None
        self.label_column: Optional[CategoricalColumnWithIdentity] = None
        self.time_column: Optional[CategoricalColumnWithIdentity] = None

        self.user_columns: List[Union[CategoricalColumnWithIdentity, NumericColumn]] = list()
        self.item_columns: List[Union[CategoricalColumnWithIdentity, NumericColumn]] = list()
        self.context_columns: List[Union[CategoricalColumnWithIdentity, NumericColumn]] = list()

        self.interaction_info: Dict = dict()

    def get_user_interaction_statistic(self, interaction_df: DataFrame) -> None:
        """补充数据集交互统计信息"""

        def create_interaction_description(interaction_count_series: Series) -> Dict:
            """某种类型交互的用户交互次数统计值"""
            interaction_description = dict()
            interaction_description[MIN] = interaction_count_series.min()
            interaction_description[MAX] = interaction_count_series.max()
            interaction_description[MEAN] = interaction_count_series.mean()
            interaction_description[MEDIAN] = interaction_count_series.median()
            return interaction_description

        # 用户交互统计
        all_user_interaction_count: Series = interaction_df.groupby(UID)[IID].count().astype(np.int32)
        self.interaction_info[ALL] = create_interaction_description(all_user_interaction_count)

        positive_user_interaction_count: Series = interaction_df[interaction_df[LABEL] == 1].groupby(UID)[
            IID].count().astype(np.int32)
        self.interaction_info[POSITIVE] = create_interaction_description(positive_user_interaction_count)

        negative_user_interaction_count: Series = interaction_df[interaction_df[LABEL] == 0].groupby(UID)[
            IID].count().astype(np.int32)
        self.interaction_info[NEGATIVE] = create_interaction_description(negative_user_interaction_count)

    def __str__(self):
        s = f"Info: {self.info}\n"
        s += "\n"
        s += f"UID Column: {self.uid_column}\n"
        s += f"IID Column: {self.iid_column}\n"
        s += f"Rate Column: {self.rate_column}\n"
        s += f"Label Column: {self.label_column}\n"
        s += f"Time Column: {self.time_column}\n"
        s += "\n"
        s += f"Context Columns:\n"
        for column in self.context_columns:
            s += f"\t{column}\n"
        s += "\n"
        s += f"User Columns:\n"
        for column in self.user_columns:
            s += f"\t{column}\n"
        s += "\n"
        s += f"Item Columns:\n"
        for column in self.item_columns:
            s += f"\t{column}\n"
        s += "\n"
        for key, d in self.interaction_info.items():
            s += f"Interaction {key}:\n"
            for dk, dv in d.items():
                s += f"\t{dk}: {dv}\n"
        return s

    def to_txt_file(self, filename: str):
        """输出到文件"""
        with open(filename, "w") as file:
            file.write(self.__str__())
