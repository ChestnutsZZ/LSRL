from typing import List, Dict, Any

import numpy as np
from numpy import ndarray
from numpy.random import default_rng  # noqa

from torchrec.data.SimpleDataReader import SimpleDataReader
from torchrec.data.dataset import SplitMode
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.task import TrainMode
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *


class HistoryDataReader(SimpleDataReader):
    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass

    def __init__(self,
                 dataset: str,
                 split_mode: SplitMode,
                 warm_n: int,
                 vt_ratio: float,
                 leave_k: int,
                 neg_sample_n: int,
                 load_feature: bool,
                 append_id: bool,
                 train_mode: TrainMode,
                 random_seed: int,
                 max_his_len: int,
                 use_neg_his: bool,
                 ):
        self.max_his_len = max_his_len
        self.use_neg_his = use_neg_his
        super().__init__(dataset, split_mode, warm_n, vt_ratio, leave_k, neg_sample_n, load_feature, append_id,
                         train_mode, random_seed)

    def _load_dataset(self) -> None:
        """数据集加载过程，构造函数调用，子类应该重载"""
        self._load_interactions()
        self._create_feature_column_dict()
        self._load_history()  # todo 会在df中加入列表，与目前的列值不相容，之后添加功能
        self._load_items()
        self._split_interactions()
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            self._load_neg_sample()
        if self.train_mode == TrainMode.PAIR_WISE:
            self._prepare_train_neg_sample()

    def _load_history(self) -> None:
        history_dir = os.path.join(DATASET_DIR, self.dataset, HISTORY_DIR)

        pos_his_npy = os.path.join(history_dir, POS_HIS_NPY_TEMPLATE % self.max_his_len)
        pos_his_mix_array: ndarray = np.load(pos_his_npy)
        assert pos_his_mix_array.shape[0] == len(self.interaction_df)
        self.interaction_df[POS_HIS_LEN] = pos_his_mix_array[:, 0].clip(min=1)
        self.interaction_df[POS_HIS] = list(pos_his_mix_array[:, 1:])

        if self.use_neg_his:
            neg_his_npy = os.path.join(history_dir, NEG_HIS_NPY_TEMPLATE % self.max_his_len)
            neg_his_mix_array: ndarray = np.load(neg_his_npy)
            assert neg_his_mix_array.shape[0] == len(self.interaction_df)
            self.interaction_df[NEG_HIS_LEN] = neg_his_mix_array[:, 0].clip(min=1)
            self.interaction_df[NEG_HIS] = list(neg_his_mix_array[:, 1:])

    def _create_feature_column_dict(self) -> None:
        """生成数据列信息"""
        super()._create_feature_column_dict()
        self.feature_column_dict[POS_HIS_LEN] = CategoricalColumnWithIdentity(category_num=0, feature_name=POS_HIS_LEN)
        self.feature_column_dict[POS_HIS] = CategoricalColumnWithIdentity(category_num=0, feature_name=POS_HIS)
        if self.use_neg_his:
            self.feature_column_dict[NEG_HIS_LEN] = CategoricalColumnWithIdentity(
                category_num=0, feature_name=NEG_HIS_LEN)
            self.feature_column_dict[NEG_HIS] = CategoricalColumnWithIdentity(category_num=0, feature_name=NEG_HIS)


if __name__ == '__main__':
    from torchrec.utils.system import init_console_logger
    from torchrec.utils.timer import Timer
    from pprint import pprint

    init_console_logger()
    with Timer():
        reader = HistoryDataReader(
            dataset='MovieLens-100K-PN',
            split_mode=SplitMode.LEAVE_K_OUT,
            warm_n=5,
            vt_ratio=0.1,
            leave_k=1,
            neg_sample_n=99,
            load_feature=False,
            append_id=False,
            train_mode=TrainMode.POINT_WISE,
            random_seed=2020,
            max_his_len=10,
            use_neg_his=True,
        )

    print(reader.get_train_dataset_size())
    print(reader.get_dev_dataset_size())
    print(reader.get_test_dataset_size())
    print(reader.get_train_dataset_item(0))
    print(reader.get_dev_dataset_item(0))
    print(reader.get_test_dataset_item(0))

    pprint(reader.get_feature_column_dict())
