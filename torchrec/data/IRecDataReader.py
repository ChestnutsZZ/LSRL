"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
from typing import Dict, Any


class IDataReader:
    """数据读入接口类"""
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息供其他模块使用"""
        raise NotImplementedError

    def get_train_dataset_size(self) -> int:
        """获取训练集大小"""
        raise NotImplementedError

    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""
        raise NotImplementedError

    def get_validation_dataset_size(self) -> int:
        """获取训练集大小"""
        raise NotImplementedError

    def get_validation_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""
        raise NotImplementedError

    def get_test_dataset_size(self) -> int:
        """获取训练集大小"""
        raise NotImplementedError

    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""
        raise NotImplementedError
