from typing import Dict, Type

from torch.nn.modules.loss import _Loss, MSELoss  # noqa

from torchrec.loss.BPRLoss import BPRLoss
from torchrec.loss.Top1Loss import Top1Loss

_loss_classes: Dict[str, Type[_Loss]] = {
    "bpr": BPRLoss,
    "top1": Top1Loss,
    "mse": MSELoss,
}

loss_name_list = _loss_classes.keys()


def get_loss(loss_name: str) -> Type[_Loss]:
    """根据loss名称获取类型"""
    if (not isinstance(loss_name, str)) or (loss_name not in _loss_classes):
        raise ValueError(f"loss_name参数不合法: {loss_name}")
    return _loss_classes[loss_name]
