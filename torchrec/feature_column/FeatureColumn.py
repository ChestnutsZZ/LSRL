"""
特征列
"""
from abc import abstractmethod, ABC
from typing import Dict, Any

from torch import Tensor


class FeatureColumn(ABC):
    """特征列基类"""

    def __init__(self):
        self.__info: Dict[str, Any] = dict()

    def set_info(self, key: str, value: Any) -> None:
        """设置额外信息"""
        self.__info[key] = value

    def get_info(self) -> Dict:
        """获取额外信息"""
        return self.__info

    @abstractmethod
    def get_feature_data(self, *args, **kwargs) -> Tensor:
        """从批量数据中获取特征数据"""
