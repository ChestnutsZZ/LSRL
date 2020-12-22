"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
from abc import abstractmethod, ABC
from typing import Dict, Any

from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.utils.argument import IWithArguments


class IDataReader(IWithArguments, ABC):
    """数据读入接口类"""

    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset
        super().__init__()

    @abstractmethod
    def train_neg_sample(self) -> None:
        """训练集负采样"""

    @abstractmethod
    def get_feature_column_dict(self) -> Dict[str, CategoricalColumnWithIdentity]:
        """获取特征列信息"""
        # todo 暂时只考虑类型列，之后应该扩展至类型列与数值列

    @abstractmethod
    def get_train_dataset_size(self) -> int:
        """获取训练集大小"""

    @abstractmethod
    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""

    @abstractmethod
    def get_dev_dataset_size(self) -> int:
        """获取验证集大小"""

    @abstractmethod
    def get_dev_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取验证集数据"""

    @abstractmethod
    def get_test_dataset_size(self) -> int:
        """获取测试集大小"""

    @abstractmethod
    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取测试集数据"""
