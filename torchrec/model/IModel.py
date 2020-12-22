"""
模型接口类
"""
import copy
import pickle
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy import ndarray
from torch.nn import Module
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Union, Optional, List, Dict

from torchrec.callback.CallbackList import CallbackList
from torchrec.callback.History import History
from torchrec.callback.ICallback import ICallback
from torchrec.data.adapter import TrainDataset
from torchrec.metric.IMetric import IMetric
from torchrec.metric.MetricList import MetricList
from torchrec.task import TrainMode
from torchrec.utils.argument import IWithArguments
from torchrec.utils.data_structure import tensor_to_device
# todo 参数相关两个函数
from torchrec.utils.global_utils import set_torch_seed


# todo 为了性能，临时删除batch回调，使用tqdm


class IModel(Module, IWithArguments, ABC):
    """模型接口类"""

    def __init__(self, random_seed: int, **kwargs):  # noqa
        set_torch_seed(random_seed)
        super().__init__()

        self.stop_training = False  # 部分回调会在这里停止训练，一旦设置为true，当前epoch结束后停止

        self.best_state_dict = None

        self.history: Optional[History] = None

        self._is_compiled = False
        self.compiled_optimizers: Optional[Optimizer] = None
        self.compiled_loss: Optional[_Loss] = None
        self.compiled_metrics: Optional[MetricList] = None
        self.compiled_device: Optional[torch.device] = None

        self._init_weights()

        self._reset_weights()

    @abstractmethod
    def _init_weights(self):
        pass

    @staticmethod
    def _reset_weights_fn(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _reset_weights(self):
        self.apply(self._reset_weights_fn)

    def load_weights(self, filepath: str, device: torch.device):
        """加载权重"""
        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)

    def save_weights(self, filepath: str):
        """保存权重"""
        torch.save(self.state_dict(), filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def get_parameters(self):
        """获取优化器优化参数，返回列表"""
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0.0}]
        return optimize_dict

    def compile(self, optimizer: Optimizer, loss: _Loss, metrics: List[IMetric], device: torch.device):
        """
        :param optimizer: 优化器，如果是个列表，必须与模型参数列表长度一致
        :param loss: 损失函数
        :param metrics: 评价指标列表
        :param device: 模型所在设备
        """
        if not isinstance(optimizer, Optimizer):
            raise ValueError(f"optimizer参数不合法: {optimizer}")
        if not isinstance(loss, _Loss):
            raise ValueError(f"loss参数不合法: {loss}")
        if (not isinstance(metrics, list)) or not all(isinstance(m, IMetric) for m in metrics):
            raise ValueError(f"metrics参数不合法: {metrics}")
        if not isinstance(device, torch.device):
            raise ValueError(f"device参数不合法: {device}")
        self.compiled_optimizers = optimizer
        self.compiled_loss = loss
        self.compiled_metrics = MetricList(metrics)
        self.compiled_device = device
        self.to(device)
        self._is_compiled = True

    def train_step(self, data: Dict):
        """训练步骤"""
        self.train()
        data = tensor_to_device(data, self.compiled_device)
        prediction, target = self(data)
        loss = self.compiled_loss(prediction, target)
        self.compiled_optimizers.zero_grad()
        loss.backward()
        self.compiled_optimizers.step(closure=None)
        return {"loss": loss}

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
        logs = {}
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            if train_mode == TrainMode.PAIR_WISE:
                dataset.train_neg_sample()
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=workers,
                                     drop_last=drop_last)
            for data in tqdm(data_loader, leave=False):
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

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def test_step(self, data):
        """验证/测试步骤"""
        self.eval()
        data = tensor_to_device(data, self.compiled_device)
        prediction, target = self(data)
        return prediction, target

    @torch.no_grad()
    def evaluate(self,
                 dataset: Dataset,
                 batch_size: int,
                 verbose: int = 1,
                 callbacks: Optional[Union[List[ICallback], CallbackList]] = None,
                 workers: int = 0):
        """验证/测试"""
        self._assert_compile_was_called()

        batches = (len(dataset) + batch_size - 1) // batch_size

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                batches=batches)

        callbacks.on_test_begin()

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers)

        predictions: List[ndarray] = []
        targets: List[ndarray] = []

        for batch, data in tqdm(enumerate(data_loader), leave=False):
            # callbacks.on_test_batch_begin(batch)
            prediction, target = self.test_step(data)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            # callbacks.on_test_batch_end(batch)

        predictions: ndarray = np.concatenate(predictions)
        targets: ndarray = np.concatenate(targets)

        # print(predictions)

        logs = self.compiled_metrics(predictions, targets)

        callbacks.on_test_end(logs)

        return logs

    def predict_step(self, data):
        """预测步骤"""
        self.eval()
        data = tensor_to_device(data, self.compiled_device)
        prediction, _ = self(data)
        return prediction

    @torch.no_grad()
    def predict(self,
                dataset: Dataset,
                batch_size: int,
                verbose: int = 0,
                callbacks: Optional[Union[List[ICallback], CallbackList]] = None,
                workers: int = 0, ):
        """预测"""
        batches = (len(dataset) + batch_size - 1) // batch_size

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                batches=batches)

        callbacks.on_predict_begin()

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers)

        predictions: List[ndarray] = []

        for batch, data in tqdm(enumerate(data_loader), leave=False):
            # callbacks.on_predict_batch_begin(batch)
            prediction = self.evaluate_step(data)
            predictions.append(prediction.detach().cpu().numpy())
            # callbacks.on_predict_batch_end(batch)

        predictions: ndarray = np.concatenate(predictions)

        callbacks.on_predict_end()

        return predictions

    def _assert_compile_was_called(self):
        if not self._is_compiled:
            raise RuntimeError('训练/测试前必须编译模型')

    def save_best_weights(self):
        self.best_state_dict = copy.deepcopy(tensor_to_device(self.state_dict(), device=torch.device("cpu")))

    def load_best_weights(self):
        """加载权重"""
        assert self.best_state_dict is not None
        self.load_state_dict(self.best_state_dict)
        self.to(self.compiled_device)
