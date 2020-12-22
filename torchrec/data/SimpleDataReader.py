"""
ID数据加载器，加载数据集ID信息
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import gc
import logging
import pickle as pkl

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng  # noqa
from pandas import DataFrame
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Set

from torchrec.data.IDataReader import IDataReader
from torchrec.data.dataset import SplitMode
from torchrec.data.process import check_dataset_info
from torchrec.data.process import check_leave_k_out_split, generate_leave_k_out_split
from torchrec.data.process import check_sequential_split, generate_sequential_split
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.task import TrainMode
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *
from torchrec.utils.enum import get_enum_values


class SimpleDataReader(IDataReader):
    """
    数据格式：
    评分/点击预测任务的训练集、验证集、测试集，TOPK推荐任务POINTWISE模式训练集：
    {
        INDEX:      int
        UID:        int
        IID:        int
        RATE:       int
        LABEL:      int
        TIME:       int
        c_c/n_XXX:  int/float  # 上下文特征，可选
        u_c/n_XXX:  int/float  # 用户特征，可选
        i_c/n_XXX:  int/float  # 物品特征，可选
    }

    TOPK推荐任务PAIRWISE模式训练集、TOPK推荐任务验证集、测试集：{
        INDEX:      int
        UID:        int
        IID:        ndarray(int)  # 训练集长度为2，验证集测试集长度为vt_sample_n
        RATE:       int
        LABEL:      int
        TIME:       int
        c_c/n_XXX:  int/float  # 上下文特征，可选
        u_c/n_XXX:  int/float  # 用户特征，可选
        i_c/n_XXX:  ndarray(int/float)  # 物品特征，可选
    }
    """

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        """获取参数描述信息"""
        argument_descriptions = super().get_argument_descriptions()
        argument_descriptions.extend([
            ArgumentDescription(name="dataset", type_=str, help_info="数据集名称",
                                legal_value_list=check_dataset_info()),
            ArgumentDescription(name="split_mode", type_=str, help_info="数据集划分模式",
                                default_value=SplitMode.LEAVE_K_OUT.value,
                                legal_value_list=get_enum_values(SplitMode)),
            ArgumentDescription(name="warm_n", type_=int, help_info="暖用户限制",
                                default_value=5,
                                lower_closed_bound=1),
            ArgumentDescription(name="vt_ratio", type_=float, help_info="顺序划分模式下验证集测试集比例",
                                default_value=0.1,
                                lower_open_bound=0.0,
                                upper_open_bound=0.5),
            ArgumentDescription(name="leave_k", type_=int, help_info="留出划分模式下验证集测试集正例个数",
                                default_value=1,
                                lower_closed_bound=1),
            ArgumentDescription(name="neg_sample_n", type_=int, help_info="验证集测试集负采样数量",
                                default_value=99,
                                lower_closed_bound=1),
            ArgumentDescription(name="load_feature", type_=bool, help_info="是否载入特征信息",
                                default_value=False),
            ArgumentDescription(name="append_id", type_=bool, help_info="特征信息中是否包含ID信息",
                                default_value=False)
        ])
        return argument_descriptions

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        """根据预设信息检查参数值"""
        super().check_argument_values(arguments)

        # 数据集划分模式
        arguments["split_mode"] = SplitMode(arguments["split_mode"])

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
                 **kwargs):  # noqa
        """
        :param dataset: 数据集名称
        """
        super().__init__(dataset)
        self.split_mode = split_mode
        self.warm_n = warm_n
        self.vt_ratio = vt_ratio
        self.leave_k = leave_k
        self.neg_sample_n = neg_sample_n
        self.load_feature = load_feature
        self.append_id = append_id
        self.train_mode = train_mode
        self.rng = default_rng(random_seed)
        self.random_seed = random_seed

        self.interaction_df: Optional[DataFrame] = None
        self.item_df: Optional[DataFrame] = None

        self.feature_column_dict: Optional[Dict[str, CategoricalColumnWithIdentity]] = None

        self.train_index_array: Optional[ndarray] = None
        self.dev_index_array: Optional[ndarray] = None
        self.test_index_array: Optional[ndarray] = None

        self.train_df: Optional[DataFrame] = None
        self.dev_df: Optional[DataFrame] = None
        self.test_df: Optional[DataFrame] = None

        self.dev_iid_topk_array: Optional[ndarray] = None
        self.test_iid_topk_array: Optional[ndarray] = None

        self.min_iid_array_index: Optional[int] = None
        self.max_iid_array_index: Optional[int] = None
        self.user_pos_his_set_dict: Optional[Dict[int, Set[int]]] = None
        self.train_iid_pair_array: Optional[ndarray] = None

        logging.info(f'加载{dataset}数据集...')
        self._load_dataset()
        gc.collect()
        logging.info(f'加载{dataset}数据集结束')

    def _load_dataset(self) -> None:
        """数据集加载过程，构造函数调用，子类应该重载"""
        self._load_interactions()
        self._create_feature_column_dict()
        self._load_items()
        self._split_interactions()
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            self._load_neg_sample()
        if self.train_mode == TrainMode.PAIR_WISE:
            self._prepare_train_neg_sample()

    def _load_interactions(self) -> None:
        """加载交互数据"""
        logging.info("加载用户交互数据...")
        if self.load_feature:
            interaction_feather = os.path.join(DATASET_DIR, self.dataset, INTERACTION_FEATHER)
        else:
            interaction_feather = os.path.join(DATASET_DIR, self.dataset, BASE_INTERACTION_FEATHER)
        self.interaction_df: DataFrame = pd.read_feather(interaction_feather)
        logging.debug(self.interaction_df.info())
        logging.info('交互数据大小：%d' % len(self.interaction_df))
        logging.info('交互数据正负例统计：' + str(dict(self.interaction_df[LABEL].value_counts())))

    def _create_feature_column_dict(self) -> None:
        """生成数据列信息"""
        self.feature_column_dict: Dict[str, CategoricalColumnWithIdentity] = {}
        for column in self.interaction_df.columns:
            self.feature_column_dict[column] = CategoricalColumnWithIdentity.from_series(
                feature_name=column,
                series=self.interaction_df[column]
            )

    def _load_items(self) -> None:
        """加载物品数据"""
        logging.info("加载物品数据...")
        logging.debug("加载物品数据...")
        self.item_df: DataFrame = pd.read_feather(os.path.join(DATASET_DIR, self.dataset, ITEM_FEATHER))
        if not self.load_feature:
            self.item_df = self.item_df[[IID]]
        logging.debug(self.item_df.info())
        logging.info('物品集大小：%d' % len(self.item_df))

    def _split_interactions(self) -> None:
        """分隔交互数据"""
        logging.info('划分交互数据...')
        split_index_dir = os.path.join(DATASET_DIR, self.dataset, SPLIT_INDEX_DIR)

        if self.split_mode == SplitMode.SEQUENTIAL_SPLIT:
            if (self.warm_n, self.vt_ratio) not in check_sequential_split(self.dataset):
                logging.info(f"顺序划分_{self.warm_n}_{self.vt_ratio}不存在，划分数据...")
                generate_sequential_split(self.dataset, self.warm_n, self.vt_ratio)
            split_name = SEQUENTIAL_SPLIT_NAME_TEMPLATE % (self.warm_n, self.vt_ratio)
        else:  # self.dataset_mode == DatasetMode.LEAVE_K_OUT
            if (self.warm_n, self.leave_k) not in check_leave_k_out_split(self.dataset):
                logging.info(f"留出划分_{self.warm_n}_{self.leave_k}不存在，划分数据...")
                generate_leave_k_out_split(self.dataset, self.warm_n, self.leave_k)
            split_name = LEAVE_K_OUT_SPLIT_NAME_TEMPLATE % (self.warm_n, self.leave_k)

        self.train_index_array: ndarray = np.load(os.path.join(split_index_dir, TRAIN_INDEX_NPY_TEMPLATE % split_name))
        self.dev_index_array: ndarray = np.load(os.path.join(split_index_dir, DEV_INDEX_NPY_TEMPLATE % split_name))
        self.test_index_array: ndarray = np.load(os.path.join(split_index_dir, TEST_INDEX_NPY_TEMPLATE % split_name))

        self.train_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.train_index_array)]
        self.dev_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.dev_index_array)]
        self.test_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.test_index_array)]

        logging.debug(self.train_df.info())
        logging.info('训练集大小：%d' % len(self.train_df))
        logging.info('训练集正负例统计：' + str(dict(self.train_df[LABEL].value_counts())))
        logging.debug(self.dev_df.info())
        logging.info('验证集大小：%d' % len(self.dev_df))
        logging.info('验证集正负例统计：' + str(dict(self.dev_df[LABEL].value_counts())))
        logging.debug(self.test_df.info())
        logging.info('测试集大小：%d' % len(self.test_df))
        logging.info('测试集正负例统计：' + str(dict(self.test_df[LABEL].value_counts())))

    def _load_neg_sample(self) -> None:
        """
        加载负采样信息，合并验证集测试集的 IID 信息
        """
        logging.info('加载负采样信息...')
        neg_sample_dir = os.path.join(DATASET_DIR, self.dataset, NEGATIVE_SAMPLE_DIR)

        # 生成实际用户列表（因为暖用户可能会删去一些），注意下标-1（PAD）
        user_index_array: ndarray = self.dev_df[UID].values - 1

        logging.debug('加载验证集负采样信息...')
        dev_neg_npy = os.path.join(neg_sample_dir, DEV_NEG_NPY_TEMPLATE % (self.random_seed, self.neg_sample_n))
        print(np.load(dev_neg_npy).shape)
        dev_neg_array: ndarray = np.load(dev_neg_npy)[user_index_array]
        dev_pos_array: ndarray = self.dev_df[IID].values.reshape(-1, 1)
        self.dev_iid_topk_array: ndarray = np.hstack((dev_pos_array, dev_neg_array))
        logging.debug(self.dev_iid_topk_array)
        logging.debug(self.dev_iid_topk_array.shape)

        logging.debug('加载测试集负采样信息...')
        test_neg_npy = os.path.join(neg_sample_dir, TEST_NEG_NPY_TEMPLATE % (self.random_seed, self.neg_sample_n))
        test_neg_array: ndarray = np.load(test_neg_npy)[user_index_array]
        test_pos_array: ndarray = self.test_df[IID].values.reshape(-1, 1)
        self.test_iid_topk_array: ndarray = np.hstack((test_pos_array, test_neg_array))
        logging.debug(self.test_iid_topk_array)
        logging.debug(self.test_iid_topk_array.shape)

        assert self.dev_iid_topk_array.shape[1] == self.test_iid_topk_array.shape[1]

    def _prepare_train_neg_sample(self) -> None:
        """加载必要历史信息，为训练集负采样做准备工作"""
        logging.info('训练集负采样准备工作...')

        self.min_iid_array_index = 1  # pad: 0
        self.max_iid_array_index = self.item_df[IID].max() + 1

        logging.debug("删除训练集中的负例...")
        self.train_df: DataFrame = self.train_df[self.train_df[LABEL] == 1]
        logging.debug(self.train_df)
        logging.info('删去负例后训练集大小：%d' % len(self.train_df))
        logging.info('删去负例后训练集正负例统计：' + str(dict(self.train_df[LABEL].value_counts())))

        logging.debug('读入交互历史统计信息...')
        neg_sample_dir = os.path.join(DATASET_DIR, self.dataset, NEGATIVE_SAMPLE_DIR)
        with open(os.path.join(neg_sample_dir, USER_POS_HIS_SET_DICT_PKL), 'rb') as user_pos_his_set_dict_pkl:
            self.user_pos_his_set_dict: Dict[int, Set[int]] = pkl.load(user_pos_his_set_dict_pkl)

        train_neg_array: ndarray = np.empty_like(self.train_df[IID].values).reshape(-1, 1)
        train_pos_array: ndarray = self.train_df[IID].values.reshape(-1, 1)
        self.train_iid_pair_array: ndarray = np.hstack((train_pos_array, train_neg_array))
        logging.debug(self.train_iid_pair_array.shape)

    def train_neg_sample(self) -> None:
        """训练集负采样"""
        assert self.train_mode == TrainMode.PAIR_WISE
        logging.info('训练集负采样...')
        neg_iid_array: ndarray = self.rng.integers(
            low=self.min_iid_array_index,
            high=self.max_iid_array_index,
            size=len(self.train_df.index),
            dtype=np.int32
        )
        for index, uid in tqdm(enumerate(self.train_df[UID].values), total=len(self.train_df.index), leave=False):
            # 该用户训练集中正向交互过的物品 ID 集合
            inter_iid_set = self.user_pos_his_set_dict[uid]
            while neg_iid_array[index] in inter_iid_set:
                neg_iid_array[index] = self.rng.integers(
                    low=self.min_iid_array_index,
                    high=self.max_iid_array_index,
                    dtype=np.int32
                )
        self.train_iid_pair_array[:, 1] = neg_iid_array
        logging.info('训练集负采样结束')

    def get_feature_column_dict(self) -> Dict[str, CategoricalColumnWithIdentity]:
        """获取特征列信息"""
        return self.feature_column_dict

    def get_train_dataset_size(self) -> int:
        """训练集大小"""
        return len(self.train_df.index)

    def get_dev_dataset_size(self) -> int:
        """验证集大小"""
        return len(self.dev_df.index)

    def get_test_dataset_size(self) -> int:
        """测试集大小"""
        return len(self.test_df.index)

    @staticmethod
    def dataframe_to_dict(df: DataFrame) -> Dict[str, Any]:
        """DataFrame转化为dict"""
        return {column: df[column].values for column in df}

    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个训练集信息"""
        train_dataset_item = dict(self.train_df.iloc[index])
        train_dataset_item[INDEX] = index
        if self.train_mode == TrainMode.PAIR_WISE:
            items_dict = self.dataframe_to_dict(self.item_df.iloc[self.train_iid_pair_array[index] - 1])  # 0: PAD
            for key in items_dict:
                train_dataset_item[key] = items_dict[key]
        return train_dataset_item

    def get_dev_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个验证集信息"""
        dev_dataset_item = dict(self.dev_df.iloc[index])
        dev_dataset_item[INDEX] = index
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            items_dict = self.dataframe_to_dict(self.item_df.iloc[self.dev_iid_topk_array[index] - 1])  # 0: PAD
            for key in items_dict:
                dev_dataset_item[key] = items_dict[key]
        return dev_dataset_item

    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个测试集信息"""
        test_dataset_item = dict(self.test_df.iloc[index])
        test_dataset_item[INDEX] = index
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            items_dict = self.dataframe_to_dict(self.item_df.iloc[self.test_iid_topk_array[index] - 1])  # 0: PAD
            for key in items_dict:
                test_dataset_item[key] = items_dict[key]
        return test_dataset_item


if __name__ == '__main__':
    from torchrec.utils.system import init_console_logger
    from torchrec.utils.timer import Timer
    from pprint import pprint

    init_console_logger()
    with Timer():
        reader = SimpleDataReader(
            dataset='MovieLens-100K-PN',
            split_mode=SplitMode.LEAVE_K_OUT,
            warm_n=5,
            vt_ratio=0.1,
            leave_k=1,
            neg_sample_n=99,
            load_feature=True,
            append_id=True,
            train_mode=TrainMode.PAIR_WISE,
            random_seed=2020
        )

    with Timer(divided_by=2):
        for i in range(2):
            reader.train_neg_sample()

    print(reader.get_train_dataset_size())
    print(reader.get_dev_dataset_size())
    print(reader.get_test_dataset_size())
    print(reader.get_train_dataset_item(0))
    print(reader.get_dev_dataset_item(0))
    print(reader.get_test_dataset_item(0))

    pprint(reader.get_feature_column_dict())
