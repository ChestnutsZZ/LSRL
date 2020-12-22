from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from typing import Dict, Type

from torchrec.optim.AdamW import AdamW

_optimizer_classes: Dict[str, Type[Optimizer]] = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}

optimizer_name_list = _optimizer_classes.keys()


def get_optimizer(optimizer_name: str) -> Type[Optimizer]:
    """根据优化器名称获取类型"""
    if (not isinstance(optimizer_name, str)) or (optimizer_name not in _optimizer_classes):
        raise ValueError(f"optimizer_name参数不合法: {optimizer_name}")
    return _optimizer_classes[optimizer_name]
