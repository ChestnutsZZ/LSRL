import torch
from torch import Tensor


def get_valid_his_index(his_ids: Tensor) -> Tensor:
    valid_his_index = his_ids.gt(0).byte()  # [batch_size, max_his_len]
    # 确保至少有一个历史，避免用户的第一个交互因为没有历史引发数值错误
    # 此外，这个PAD物品可以作为没有交互初始物品表示
    valid_his_index[:, 0] = 1
    return valid_his_index


def get_postion_ids(valid_ids: Tensor, seq_len: Tensor, device: torch.device) -> Tensor:
    batch_size, max_seq_len = valid_ids.shape
    position = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, max_his_len]
    position = (seq_len.unsqueeze(-1) - position) * valid_ids.long()  # [batch_size, max_his_len]
    return position
