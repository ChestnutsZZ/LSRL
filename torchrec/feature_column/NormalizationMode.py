"""
数值列标准化模式
"""
from enum import Enum, unique


@unique
class NormalizationMode(Enum):
    """数值列标准化模式"""
    NOP = "nop"
    MAX_MIN = "max_min"
    Z_SCORE = "z_score"
