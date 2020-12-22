"""
枚举类相关辅助功能
"""
from enum import Enum
from typing import Type, List


def get_enum_values(enum_t: Type[Enum]) -> List:
    """获取某枚举子类所有原始值"""
    return [member.value for member in enum_t]
