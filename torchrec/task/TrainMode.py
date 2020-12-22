"""
训练模式
"""
from enum import Enum, unique


@unique
class TrainMode(Enum):
    """训练模式"""
    POINT_WISE = "point_wise"
    PAIR_WISE = "pair_wise"
