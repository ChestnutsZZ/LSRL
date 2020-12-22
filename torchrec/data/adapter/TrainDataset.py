"""
训练集适配器
"""
from torch.utils.data import Dataset

from torchrec.data.IDataReader import IDataReader


class TrainDataset(Dataset):
    """训练集适配器"""

    def __init__(self, data_reader: IDataReader):
        self.data_reader: IDataReader = data_reader

    def __len__(self):
        return self.data_reader.get_train_dataset_size()

    def __getitem__(self, item):
        return self.data_reader.get_train_dataset_item(item)

    def train_neg_sample(self):
        self.data_reader.train_neg_sample()
