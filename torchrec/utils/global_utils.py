"""
全局辅助函数
"""
import torch


def set_torch_seed(seed: int) -> None:
    """
    设置随机种子保证torch可以重现
    :param seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # noqa
    torch.cuda.manual_seed_all(seed)  # noqa
    torch.backends.cudnn.deterministic = True  # noqa
    torch.backends.cudnn.benchmark = False  # noqa
