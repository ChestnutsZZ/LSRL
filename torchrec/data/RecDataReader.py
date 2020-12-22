"""
ID数据加载器，加载数据集ID信息
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import gc
import json
import logging
import pickle as pkl
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng
from pandas import DataFrame
from tqdm import tqdm

from data.data_reader.IRecDataReader import IRecDataReader
from global_utils.argument import IWithArguments, ArgumentDescription
from global_utils.const import *


class RecDataReader(IRecDataReader, IWithArguments):
    """
    数据格式：
    评分/点击预测任务的训练集、验证集、测试集，TOPK推荐任务POINTWISE模式训练集：{
        INDEX:      np.int32
        UID:        np.int32
        UCFEATURE:  ndarray(np.int32)  # 根据参数可选，根据参数决定是否加入UID信息
        IID:        np.int32
        ICEATURE:   ndarray(np.int32)  # 根据参数可选，根据参数决定是否加入IID信息
        LABEL:      np.int32
    }
    TOPK推荐任务PAIRWISE模式训练集、TOPK推荐任务验证集、测试集：{
        INDEX:      np.int32
        UID:        np.int32
        UCFEATURE:  ndarray(np.int32)  # 根据参数可选，根据参数决定是否加入UID信息
        IID:        ndarray(np.int32)  # 训练集长度为2，验证集测试集长度为vt_sample_n
        ICFEATURE:  ndarray(np.int32)  # 根据参数可选，根据参数决定是否加入IID信息，2维数组，第一维训练集长度为2，验证集测试集长度为vt_sample_n
        LABEL:      np.int32
    }

    子类根据需要重新实现函数：
    """

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        """获取参数描述信息"""
        return [
            ArgumentDescription(name="dataset", type_=str, is_required=True, help_info="数据集名称"),
            ArgumentDescription(name="load_feature", type_=int, is_required=True, help_info="是否载入特征信息"),
            ArgumentDescription(name="append_id", type_=int, is_required=True, help_info="特征信息中是否包含ID信息")
        ]

    @classmethod
    def check_argument_values(cls, args: Dict[str, Any]) -> None:
        """检查参数值"""
        assert args["load_feature"] in {0, 1}
        assert args["append_id"] in {0, 1}

    def __init__(self, dataset: str, load_feature: int, append_id: int, train_mode: str, task_mode: str,
                 random_seed: int):
        """
        :param dataset: 数据集名称
        """
        self.dataset = dataset
        self.load_feature = load_feature
        self.append_id = append_id
        self.train_mode = train_mode
        self.task_mode = task_mode
        self.rng = default_rng(random_seed)
        logging.info(f'加载{dataset}数据集...')
        self.process()
        self.after_process()
        self.pandas_to_numpy()
        gc.collect()
        logging.info(f'加载{dataset}数据集结束')

    def process(self) -> None:
        """数据加载并处理"""
        self.__load_train_validation_test_data()
        if self.load_feature == 1:
            self.__load_user_item_data()
        self.__load_dataset_info()

    def __load_train_validation_test_data(self) -> None:
        """载入训练验证测试集"""
        logging.info('加载训练集...')
        train_pkl = os.path.join(DATASET_DIR, self.dataset, TRAIN_PKL)
        self.train_df: DataFrame = pd.read_pickle(train_pkl)
        if self.load_feature == 0:
            self.train_df.drop(columns=[column for column in self.train_df.columns if column.startswith("c_")],
                               inplace=True)
        logging.info('训练集大小：%d' % len(self.train_df))
        logging.info('训练集正负例统计：' + str(dict(self.train_df[LABEL].value_counts())))

        logging.info('加载验证集...')
        validation_pkl = os.path.join(DATASET_DIR, self.dataset, VALIDATION_PKL)
        self.validation_df: DataFrame = pd.read_pickle(validation_pkl)
        if self.load_feature == 0:
            self.validation_df.drop(
                columns=[column for column in self.validation_df.columns if column.startswith("c_")], inplace=True
            )
        logging.info('验证集大小：%d' % len(self.validation_df))
        logging.info('验证集正负例统计：' + str(dict(self.validation_df[LABEL].value_counts())))

        logging.info('加载测试集...')
        test_pkl = os.path.join(DATASET_DIR, self.dataset, TEST_PKL)
        self.test_df: DataFrame = pd.read_pickle(test_pkl)
        if self.load_feature == 0:
            self.test_df.drop(columns=[column for column in self.test_df.columns if column.startswith("c_")],
                              inplace=True)
        logging.info('测试集大小：%d' % len(self.test_df))
        logging.info('测试集正负例统计：' + str(dict(self.test_df[LABEL].value_counts())))

    def __load_user_item_data(self) -> None:
        """加载用户与物品信息"""
        logging.info('加载用户信息...')
        user_pkl = os.path.join(DATASET_DIR, self.dataset, USER_PKL)
        self.user_df: DataFrame = pd.read_pickle(user_pkl)
        if self.append_id == 0:
            self.user_df.drop(columns=UID, inplace=True)
        logging.info('用户集大小：%d' % len(self.user_df))

        logging.info('加载物品信息...')
        item_pkl = os.path.join(DATASET_DIR, self.dataset, ITEM_PKL)
        self.item_df: DataFrame = pd.read_pickle(item_pkl)
        if self.append_id == 0:
            self.item_df.drop(columns=IID, inplace=True)
        logging.info('物品集大小：%d' % len(self.item_df))

    def __load_dataset_info(self) -> None:
        """加载数据集统计信息"""
        logging.info('加载数据集统计信息...')
        max_info_json = os.path.join(DATASET_DIR, self.dataset, STATISTIC_DIR, MAX_INFO_JSON)
        with open(max_info_json) as max_info_f:
            self.column_max: dict = json.loads(max_info_f.read())
        min_info_json = os.path.join(DATASET_DIR, self.dataset, STATISTIC_DIR, MIN_INFO_JSON)
        with open(min_info_json) as min_info_f:
            self.column_min: dict = json.loads(min_info_f.read())
        self.label_max: int = self.column_max[LABEL]
        self.label_min: int = self.column_min[LABEL]
        logging.info(f'标签范围：{self.label_min} - {self.label_max}')
        self.user_num: int = self.column_max[UID] + 1
        self.item_num: int = self.column_max[IID] + 1
        logging.info(f'用户（含PAD0）数量：{self.user_num}')
        logging.info(f'物品（含PAD0）数量：{self.item_num}')

    # noinspection PyAttributeOutsideInit
    def after_process(self) -> None:
        """数据加载后的钩子函数"""
        if self.task_mode == TOPK:
            self.__drop_neg_data_in_validation_test_data()
            self.__load_and_merge_neg_sample_into_validation_test_data()
            if self.train_mode == PAIR_WISE:
                self.__drop_neg_data_in_train_data()
                self.__load_user_his_info_and_prepare_train_neg_sample()

    def __drop_neg_data_in_validation_test_data(self) -> None:
        """TopK任务中，删除原有验证测试集的负样例，该函数需要在after_process中调用，保证不影响process中的处理"""
        self.validation_df = self.validation_df[self.validation_df[LABEL] > 0].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df[LABEL] > 0].reset_index(drop=True)

    def __load_and_merge_neg_sample_into_validation_test_data(self) -> None:
        """
        加载负采样信息，合并验证集测试集的 IID 信息，该函数需要在after_process中__drop_neg_data_in_validation_test_data函数后调用
        """
        logging.info('加载验证集负采样信息...')
        validation_neg_npy = os.path.join(DATASET_DIR, self.dataset, NEGATIVE_SAMPLE_DIR, VALIDATION_NEG_NPY)
        validation_neg_array: ndarray = np.load(validation_neg_npy)
        validation_pos_array: ndarray = self.validation_df[IID].values.reshape(-1, 1)
        self.validation_iid_topk_array: ndarray = np.hstack((validation_pos_array, validation_neg_array))

        logging.info('加载测试集负采样信息...')
        test_neg_npy = os.path.join(DATASET_DIR, self.dataset, NEGATIVE_SAMPLE_DIR, TEST_NEG_NPY)
        test_neg_array: ndarray = np.load(test_neg_npy)
        test_pos_array: ndarray = self.test_df[IID].values.reshape(-1, 1)
        self.test_iid_topk_array: ndarray = np.hstack((test_pos_array, test_neg_array))

        assert self.validation_iid_topk_array.shape[1] == self.test_iid_topk_array.shape[1]
        self.vt_sample_n = self.test_iid_topk_array.shape[1]
        logging.info(f'验证集测试集负采样数量：{self.vt_sample_n - 1}')

    def __drop_neg_data_in_train_data(self) -> None:
        """TopK任务且pair-wise训练模式中，删除原有测试集的负样例，该函数需要在after_process中调用，保证不影响process中的处理"""
        self.train_df = self.train_df[self.train_df[LABEL] > 0].reset_index(drop=True)

    def __load_user_his_info_and_prepare_train_neg_sample(self) -> None:
        """加载必要历史信息，为训练集负采样做准备工作，该函数需要在after_process中__drop_neg_data_in_train_data函数后调用"""
        logging.info('训练集负采样准备工作...')
        logging.info('获得训练集中所有用户交互过的物品ID集合...')
        self.train_all_pos_iid_array: ndarray = np.array(sorted(list(set(self.train_df[IID].values))), dtype=np.int32)
        self.train_max_pos_iid_array_index = len(self.train_all_pos_iid_array)
        logging.info(f'训练集中交互过的物品数：{self.train_max_pos_iid_array_index}')

        logging.info('读入交互历史信息...')
        train_pos_his_set_dict_pkl = os.path.join(DATASET_DIR, self.dataset, STATISTIC_DIR,
                                                  TRAIN_USER_POS_HIS_SET_DICT_PKL)
        with open(train_pos_his_set_dict_pkl, 'rb') as train_pos_his_set_dict_f:
            self.train_pos_his_dict = pkl.load(train_pos_his_set_dict_f)

        train_neg_array: ndarray = np.empty_like(self.train_df[UID].values).reshape(-1, 1)
        train_pos_array: ndarray = self.train_df[IID].values.reshape(-1, 1)
        self.train_iid_topk_array: ndarray = np.hstack((train_pos_array, train_neg_array))

    # noinspection PyAttributeOutsideInit
    def pandas_to_numpy(self) -> None:
        """关键数据从 DataFrame 类转换到 ndarray 类提高速度"""
        logging.info('利用numpy矩阵加速...')
        self.train_uid_array: ndarray = self.train_df[UID].values
        self.train_iid_array: ndarray = self.train_df[IID].values
        self.train_label_array: ndarray = self.train_df[LABEL].values
        self.validation_uid_array: ndarray = self.validation_df[UID].values
        self.validation_iid_array: ndarray = self.validation_df[IID].values
        self.validation_label_array: ndarray = self.validation_df[LABEL].values
        self.test_uid_array: ndarray = self.test_df[UID].values
        self.test_iid_array: ndarray = self.test_df[IID].values
        self.test_label_array: ndarray = self.test_df[LABEL].values
        if self.load_feature:
            self.user_feature_array: ndarray = self.user_df.values
            self.item_feature_array: ndarray = self.item_df.values

    def train_neg_sample(self) -> None:
        """训练集负采样"""
        assert self.train_mode == PAIR_WISE
        logging.info('训练集负采样...')
        neg_iid_array: ndarray = self.train_all_pos_iid_array[
            self.rng.integers(self.train_max_pos_iid_array_index, size=len(self.train_df.index))
        ]
        for index, uid in tqdm(enumerate(self.train_df[UID].values), total=len(self.train_df.index), leave=False):
            # 该用户训练集中正向交互过的物品 ID 集合
            inter_iid_set = self.train_pos_his_dict[uid]
            while neg_iid_array[index] in inter_iid_set:
                neg_iid_array[index] = self.train_all_pos_iid_array[
                    self.rng.integers(self.train_max_pos_iid_array_index)]
        self.train_iid_topk_array[:, 1] = neg_iid_array
        logging.info('训练集负采样结束')

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        返回数据集其他可能会被模型或训练器使用的信息
        :return 信息字典
        """
        return {
            'label_max': self.label_max,
            'label_min': self.label_min,
            'user_num': self.user_num,
            'item_num': self.item_num,
            'vt_sample_n': self.vt_sample_n if self.task_mode == TOPK else None
        }

    def get_train_dataset_size(self) -> int:
        """训练集大小"""
        return len(self.train_df.index)

    def get_validation_dataset_size(self) -> int:
        """验证集大小"""
        return len(self.validation_df.index)

    def get_test_dataset_size(self) -> int:
        """测试集大小"""
        return len(self.test_df.index)

    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个训练集信息"""
        train_dataset_item = {
            INDEX: index,
            UID: self.train_uid_array[index],
            IID: self.train_iid_array[index] if self.train_mode == POINT_WISE else self.train_iid_topk_array[index],
            LABEL: self.train_label_array[index]
        }
        if self.load_feature:
            train_dataset_item[UCFEATURE] = self.user_feature_array[self.train_uid_array[index] - 1]
            if self.train_mode == POINT_WISE:
                train_dataset_item[ICFEATURE] = self.item_feature_array[self.train_iid_array[index] - 1]
            else:
                train_dataset_item[ICFEATURE] = self.item_feature_array[self.train_iid_topk_array[index] - 1]
        return train_dataset_item

    def get_validation_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个验证集信息"""
        validation_dataset_item = {
            INDEX: index,
            UID: self.validation_uid_array[index],
            IID: self.validation_iid_array[index] if self.task_mode == PRED else self.validation_iid_topk_array[index],
            LABEL: self.validation_label_array[index]
        }
        if self.load_feature:
            validation_dataset_item[UCFEATURE] = self.user_feature_array[self.validation_uid_array[index] - 1]
            if self.task_mode == PRED:
                validation_dataset_item[ICFEATURE] = self.item_feature_array[self.validation_iid_array[index] - 1]
            else:
                validation_dataset_item[ICFEATURE] = self.item_feature_array[self.validation_iid_topk_array[index] - 1]
        return validation_dataset_item

    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个测试集信息"""
        test_dataset_item = {
            INDEX: index,
            UID: self.test_uid_array[index],
            IID: self.test_iid_array[index] if self.task_mode == PRED else self.test_iid_topk_array[index],
            LABEL: self.test_label_array[index]
        }
        if self.load_feature:
            test_dataset_item[UCFEATURE] = self.user_feature_array[self.test_uid_array[index] - 1]
            if self.task_mode == PRED:
                test_dataset_item[ICFEATURE] = self.item_feature_array[self.test_iid_array[index] - 1]
            else:
                test_dataset_item[ICFEATURE] = self.item_feature_array[self.test_iid_topk_array[index] - 1]
        return test_dataset_item


if __name__ == '__main__':
    from global_utils.global_utils import init_console_logger, Timer

    init_console_logger()
    with Timer():
        reader = RecDataReader(dataset='MovieLens-100K_5_938_932', load_feature=0, append_id=1,
                               train_mode=POINT_WISE, task_mode=PRED, random_seed=2020)
    with Timer():
        if reader.train_mode == PAIR_WISE:
            reader.train_neg_sample()
        print(reader.get_train_dataset_item(345))
        print(reader.get_validation_dataset_item(345))
        print(reader.get_test_dataset_item(345))
    with Timer():
        reader = RecDataReader(dataset='Xing_5_233996_100000', load_feature=0, append_id=1,
                               train_mode=POINT_WISE, task_mode=PRED, random_seed=2020)
    with Timer():
        if reader.train_mode == PAIR_WISE:
            reader.train_neg_sample()
        print(reader.get_train_dataset_item(345))
        print(reader.get_validation_dataset_item(345))
        print(reader.get_test_dataset_item(345))
