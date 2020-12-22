from typing import List, Dict, Any

from numpy import ndarray
from numpy.random import default_rng  # noqa

from torchrec.data.SimpleDataReader import SimpleDataReader
from torchrec.data.dataset import SplitMode
from torchrec.data.process.interaction_history_list import pad_or_cut_array
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.task import TrainMode
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *


class SVDPPDataReader(SimpleDataReader):
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
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        """根据预设信息检查参数值"""
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
                 limit: int,
                 ):  # noqa
        self.limit = limit
        self.train_all_his_dict: Dict[int, ndarray] = {}
        super().__init__(dataset, split_mode, warm_n, vt_ratio, leave_k, neg_sample_n, load_feature, append_id,
                         train_mode, random_seed)

    def _load_dataset(self) -> None:
        """数据集加载过程，构造函数调用，子类应该重载"""
        self._load_interactions()
        self._create_feature_column_dict()
        self._load_items()
        self._split_interactions()
        self._create_user_all_history()
        if self.split_mode == SplitMode.LEAVE_K_OUT:
            self._load_neg_sample()
        if self.train_mode == TrainMode.PAIR_WISE:
            self._prepare_train_neg_sample()

    def _create_feature_column_dict(self) -> None:
        """生成数据列信息"""
        super()._create_feature_column_dict()
        self.feature_column_dict[IIDS] = CategoricalColumnWithIdentity(category_num=0, feature_name=IIDS)

    def _create_user_all_history(self):
        """生成SVDPP需要的历史信息"""
        for uid, uid_df in self.train_df.groupby(UID):
            train_all_his = uid_df[IID].values
            self.train_all_his_dict[uid] = train_all_his
        for uid in self.train_all_his_dict:
            self.train_all_his_dict[uid] = pad_or_cut_array(self.train_all_his_dict[uid], self.limit)

    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个训练集信息"""
        train_dataset_item = super().get_train_dataset_item(index)
        train_dataset_item[IIDS] = self.train_all_his_dict[train_dataset_item[UID]]
        return train_dataset_item

    def get_dev_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个验证集信息"""
        dev_dataset_item = super().get_dev_dataset_item(index)
        dev_dataset_item[IIDS] = self.train_all_his_dict[dev_dataset_item[UID]]
        return dev_dataset_item

    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个测试集信息"""
        test_dataset_item = super().get_test_dataset_item(index)
        test_dataset_item[IIDS] = self.train_all_his_dict[test_dataset_item[UID]]
        return test_dataset_item


if __name__ == '__main__':
    from torchrec.utils.system import init_console_logger
    from torchrec.utils.timer import Timer
    from pprint import pprint

    init_console_logger()
    with Timer():
        reader = SVDPPDataReader(
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
            limit=256,
        )

    print(reader.get_train_dataset_size())
    print(reader.get_dev_dataset_size())
    print(reader.get_test_dataset_size())
    print(reader.get_train_dataset_item(0))
    print(reader.get_dev_dataset_item(0))
    print(reader.get_test_dataset_item(0))

    pprint(reader.get_feature_column_dict())
