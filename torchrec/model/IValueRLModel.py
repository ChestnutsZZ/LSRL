"""
模型接口类
"""
import copy
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, List, Type, Dict

from torchrec.callback.CallbackList import CallbackList
from torchrec.callback.History import History
from torchrec.callback.ICallback import ICallback
from torchrec.data.adapter import TrainDataset
from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.metric import IMetric
from torchrec.model import IModel
from torchrec.task import TrainMode


# todo 参数相关两个函数


# todo 为了性能，临时删除batch回调，使用tqdm


class IQNet(Module, ABC):
    def __init__(self):
        super().__init__()
        self.compile_device: Optional[torch.device] = None

    @abstractmethod
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        pass

    @abstractmethod
    def next_forward(self, data: Dict[str, Tensor]) -> Tensor:
        pass

    @abstractmethod
    def load_pretrain_embedding(self) -> None:
        pass


class IValueRLModel(IModel, ABC):
    """模型接口类"""

    def __init__(self,
                 random_seed: int,
                 update_freq: int,
                 gamma: float,
                 reward_column: CategoricalColumnWithIdentity,
                 q_net_type: Type[IQNet],
                 **kwargs,
                 ):
        self.update_freq = update_freq
        self.gamma = gamma
        self.reward_column = reward_column
        self.q_net_type = q_net_type
        self.q_net_params = kwargs
        super().__init__(random_seed)

    def _init_weights(self):
        self.eval_net = self.q_net_type(**self.q_net_params)  # noqa
        self.target_net = self.q_net_type(**self.q_net_params)  # noqa

    def _update_target_net(self):
        self.target_net.load_state_dict(copy.deepcopy(self.eval_net.state_dict()))

    def _reset_weights(self):
        self.eval_net.apply(self._reset_weights_fn)
        self.eval_net.load_pretrain_embedding()
        self._update_target_net()

    def get_parameters(self):
        """获取优化器优化参数，返回列表"""
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.eval_net.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0.0}]
        return optimize_dict

    def compile(self, optimizer: Optimizer, loss: _Loss, metrics: List[IMetric], device: torch.device):
        super().compile(optimizer, loss, metrics, device)
        self.eval_net.compile_device = device
        self.target_net.compile_device = device

    def fit(self,
            dataset: TrainDataset,
            batch_size: int,
            epochs: int,
            dev_dataset: Optional[Dataset],
            train_mode: TrainMode,
            verbose: int = 1,
            callbacks: Optional[Union[List[ICallback], CallbackList]] = None,
            shuffle: bool = True,
            workers: int = 0,
            drop_last: bool = False,
            dev_batch_size: Optional[int] = None,
            dev_freq: int = 1,
            ) -> History:
        """
        训练过程
        :param dataset: 训练集
        :param batch_size: 批次大小
        :param epochs: 轮次数
        :param dev_dataset: 验证集
        :param train_mode: 训练模式
        :param verbose: 0：不显示/1：进度条/2：每轮次显示最终结果
        :param callbacks: 回调列表
        :param shuffle: 是否打乱训练集
        :param workers: 数据加载进程数，0使用主进程
        :param drop_last: 是否丢到最后不足batch的数据
        :param dev_batch_size: 验证集批次大小，默认为batch_size
        :param dev_freq: 验证频率
        :return: 训练历史回调
        """
        self._assert_compile_was_called()

        if drop_last:
            batches = len(dataset) // batch_size
        else:
            batches = (len(dataset) + batch_size - 1) // batch_size

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,  # ProgbarLogger
                epochs=epochs,  # ProgbarLogger
                batches=batches  # ProgbarLogger
            )

        self.stop_training = False
        callbacks.on_train_begin()
        # batch = 0
        # logs = {}
        epoch = 0
        for epoch_index in range(epochs):
            if train_mode == TrainMode.PAIR_WISE:
                dataset.train_neg_sample()
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=workers,
                                     drop_last=drop_last)
            for data in data_loader:
                callbacks.on_epoch_begin(epoch)
                # callbacks.on_train_batch_begin(batch)
                logs = self.train_step(data)
                # callbacks.on_train_batch_end(batch, logs)
                epoch_logs = copy.copy(logs)

                if (epoch + 1) % dev_freq == 0:
                    dev_logs = self.evaluate(
                        dataset=dev_dataset,
                        batch_size=dev_batch_size or batch_size,
                        verbose=verbose,
                        callbacks=callbacks,
                        workers=workers
                    )
                    epoch_logs.update(dev_logs)

                # todo 目前按照epoch更新
                if (epoch + 1) % self.update_freq == 0:
                    self._update_target_net()

                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
                if self.stop_training:
                    break
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history
