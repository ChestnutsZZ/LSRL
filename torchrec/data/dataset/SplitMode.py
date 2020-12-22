"""
数据集划分模式
"""
from enum import Enum, unique


@unique
class SplitMode(Enum):
    """数据集划分模式"""
    SEQUENTIAL_SPLIT = "sequential_split"
    LEAVE_K_OUT = "leave_k_out"
