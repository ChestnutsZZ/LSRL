"""
数据集适配器
将IDataReader的实现子类适配到torch.utils.data.Dataset
"""
from torchrec.data.adapter.DevDataset import DevDataset
from torchrec.data.adapter.TestDataset import TestDataset
from torchrec.data.adapter.TrainDataset import TrainDataset
