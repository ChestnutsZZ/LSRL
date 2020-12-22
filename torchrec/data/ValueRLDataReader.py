from typing import List, Dict, Any

import numpy as np
from numpy import ndarray
from numpy.random import default_rng  # noqa

from torchrec.data.HistoryDataReader import HistoryDataReader
from torchrec.data.dataset import SplitMode
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.task import TrainMode
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *


class ValueRLDataReader(HistoryDataReader):
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
                 max_state_len: int,
                 use_neg_state: bool,
                 rl_sample_len: int,
                 ):
        self.max_next_state_len = max_state_len
        self.use_neg_next_state = use_neg_state
        self.rl_sample_len = rl_sample_len
        super().__init__(dataset, split_mode, warm_n, vt_ratio, leave_k, neg_sample_n, load_feature, append_id,
                         train_mode, random_seed, max_state_len, use_neg_state)

    def _load_dataset(self) -> None:
        """数据集加载过程，构造函数调用，子类应该重载"""
        self._load_interactions()
        self._create_feature_column_dict()
        self._load_history()  # todo 会在df中加入列表，与目前的列值不相容，之后添加功能
        self._load_next_state()  # todo 会在df中加入列表，与目前的列值不相容，之后添加功能
        self._load_next_item_sample()  # todo 会在df中加入列表，与目前的列值不相容，之后添加功能
        self._load_items()
        self._split_interactions()
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            self._load_neg_sample()
        if self.train_mode == TrainMode.PAIR_WISE:
            self._prepare_train_neg_sample()

    def _load_next_state(self) -> None:
        next_state_dir = os.path.join(DATASET_DIR, self.dataset, NEXT_STATE_DIR)

        pos_next_state_npy = os.path.join(next_state_dir, POS_NEXT_STATE_NPY_TEMPLATE % self.max_next_state_len)
        pos_next_state_mix_array: ndarray = np.load(pos_next_state_npy)
        assert pos_next_state_mix_array.shape[0] == len(self.interaction_df)
        self.interaction_df[POS_NEXT_STATE_LEN] = pos_next_state_mix_array[:, 0].clip(min=1)
        self.interaction_df[POS_NEXT_STATE] = list(pos_next_state_mix_array[:, 1:])

        if self.use_neg_next_state:
            neg_next_state_npy = os.path.join(next_state_dir, NEG_NEXT_STATE_NPY_TEMPLATE % self.max_next_state_len)
            neg_next_state_mix_array: ndarray = np.load(neg_next_state_npy)
            assert neg_next_state_mix_array.shape[0] == len(self.interaction_df)
            self.interaction_df[NEG_NEXT_STATE_LEN] = neg_next_state_mix_array[:, 0].clip(min=1)
            self.interaction_df[NEG_NEXT_STATE] = list(neg_next_state_mix_array[:, 1:])

    def _load_next_item_sample(self) -> None:
        rl_sample_dir = os.path.join(DATASET_DIR, self.dataset, RL_SAMPLE_DIR)

        rl_sample_npy = os.path.join(rl_sample_dir, RL_SAMPLE_NPY_TEMPLATE % self.rl_sample_len)
        rl_sample_array: ndarray = np.load(rl_sample_npy)
        assert rl_sample_array.shape[0] == len(self.interaction_df)
        self.interaction_df[RL_SAMPLE] = list(rl_sample_array)

    def _create_feature_column_dict(self) -> None:
        """生成数据列信息"""
        super()._create_feature_column_dict()
        self.feature_column_dict[RL_SAMPLE] = CategoricalColumnWithIdentity(
            category_num=0, feature_name=RL_SAMPLE)
        self.feature_column_dict[POS_NEXT_STATE_LEN] = CategoricalColumnWithIdentity(
            category_num=0, feature_name=POS_NEXT_STATE_LEN)
        self.feature_column_dict[POS_NEXT_STATE] = CategoricalColumnWithIdentity(
            category_num=0, feature_name=POS_NEXT_STATE)
        if self.use_neg_next_state:
            self.feature_column_dict[NEG_NEXT_STATE_LEN] = CategoricalColumnWithIdentity(
                category_num=0, feature_name=NEG_NEXT_STATE_LEN)
            self.feature_column_dict[NEG_NEXT_STATE] = CategoricalColumnWithIdentity(
                category_num=0, feature_name=NEG_NEXT_STATE)


if __name__ == '__main__':
    from torchrec.utils.system import init_console_logger
    from torchrec.utils.timer import Timer
    from pprint import pprint

    init_console_logger()
    with Timer():
        reader = ValueRLDataReader(
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
            max_state_len=10,
            use_neg_state=True,
            rl_sample_len=8,
        )

    print(reader.get_train_dataset_size())
    print(reader.get_dev_dataset_size())
    print(reader.get_test_dataset_size())
    print(reader.get_train_dataset_item(0))
    print(reader.get_dev_dataset_item(0))
    print(reader.get_test_dataset_item(0))

    pprint(reader.get_feature_column_dict())
