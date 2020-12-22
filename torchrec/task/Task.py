"""
单次任务执行
"""
from typing import List, Dict, Any

import torch
from numpy.random import default_rng  # noqa
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim.optimizer import Optimizer

from torchrec.callback.CSVLogger import CSVLogger
from torchrec.callback.EarlyStopping import EarlyStopping
from torchrec.callback.ModelCheckpoint import ModelCheckpoint
from torchrec.data.IDataReader import IDataReader
from torchrec.data.adapter import TrainDataset, DevDataset, TestDataset
from torchrec.loss.losses import loss_name_list
from torchrec.metric import IMetric
from torchrec.model import IModel
from torchrec.model.models import (
    model_name_list)
from torchrec.optim.optimizers import optimizer_name_list
from torchrec.task import TrainMode
from torchrec.task.ITask import ITask
from torchrec.utils.argument.ArgumentDescription import ArgumentDescription
from torchrec.utils.const import *
from torchrec.utils.enum import get_enum_values
from torchrec.utils.global_utils import set_torch_seed


class Task(ITask):
    """单次任务执行"""

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        argument_descriptions = super().get_argument_descriptions()
        argument_descriptions.extend([
            ArgumentDescription(name="debug", type_=bool, help_info="Debug模式（只运行，不存储任何文件）",
                                default_value=False),
            ArgumentDescription(name="gpu", type_=int, help_info="设置GPU序号，-1代表CPU",
                                default_value=-1,
                                lower_closed_bound=-1,
                                upper_open_bound=torch.cuda.device_count()),
            ArgumentDescription(name="model_name", type_=str, help_info="模型名称",
                                legal_value_list=model_name_list),
            ArgumentDescription(name="random_seed", type_=int, help_info="随机种子",
                                lower_closed_bound=0),
            ArgumentDescription(name="metrics", type_=str, help_info="评价指标：支持ndcg@k，hit@k，如果有多个，逗号分隔",
                                default_value="ndcg@10"),
            ArgumentDescription(name="train_mode", type_=str, help_info="训练模式",
                                default_value=TrainMode.POINT_WISE.value,
                                legal_value_list=get_enum_values(TrainMode)),
            ArgumentDescription(name="epoch", type_=int, help_info="训练轮数",
                                default_value=100,
                                lower_closed_bound=1),
            ArgumentDescription(name="batch_size", type_=int, help_info="批次大小",
                                default_value=128,
                                lower_closed_bound=1),
            ArgumentDescription(name="optimizer", type_=str, help_info="优化器种类",
                                legal_value_list=optimizer_name_list),
            ArgumentDescription(name="lr", type_=float, help_info="学习率",
                                default_value=1e-3,
                                lower_open_bound=0),
            ArgumentDescription(name="l2", type_=float, help_info="L2正则化",
                                default_value=0,
                                lower_closed_bound=0),
            ArgumentDescription(name="loss", type_=str, help_info="损失函数",
                                legal_value_list=loss_name_list),
            # todo 添加callbacks参数
            ArgumentDescription(name="num_workers", type_=int, help_info="数据加载器多进程数",
                                default_value=0,
                                lower_closed_bound=0),
            ArgumentDescription(name="dev_freq", type_=int, help_info="验证频率",
                                default_value=1,
                                lower_closed_bound=1),
        ])
        return argument_descriptions

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        """需要在数据载入类后边检查"""
        super().check_argument_values(arguments)
        # 评价指标
        # todo 根据数据载入类检查
        arguments["metrics"] = arguments["metrics"].split(",")

        def _check_metrics_name(name):
            if "@" not in name:
                return False
            m, k = name.split("@")
            if m not in {"ndcg", "hit"} or int(k) <= 0:
                return False
            return True

        assert all(map(_check_metrics_name, arguments["metrics"])), arguments["metrics"]

        # 训练模式
        arguments["train_mode"] = TrainMode(arguments["train_mode"])

    @classmethod
    def create_from_console(cls):
        pass

    #     """从控制台创建任务"""
    #     init_console_logger(logging.INFO)
    #
    #     # 获取模型类型
    #     init_parser = argparse.ArgumentParser(add_help=False)
    #     init_parser.add_argument('--model_name', type=str, help='模型名称', choices=model_name_list)
    #     init_args, init_extras = init_parser.parse_known_args()
    #     logging.info(init_args.__dict__)
    #     logging.info(init_extras)
    #
    #     # 获取数据载入器类型
    #     model_type: Type[IModel] = get_model_type(init_args.model_name)
    #     data_reader_type: Type[IDataReader] = get_data_reader_type(model_type)
    #
    #     # 解析参数
    #     parser = argparse.ArgumentParser(add_help=False)
    #     argument_descriptions = (
    #             data_reader_type.get_argument_descriptions()
    #             + model_type.get_argument_descriptions()
    #             + cls.get_argument_descriptions()
    #     )
    #     for description in argument_descriptions:
    #         description.add_argument_into_argparser(parser)
    #     origin_args, extras = parser.parse_known_args()
    #     arguments = origin_args.__dict__
    #
    #     data_reader_type.check_argument_values(arguments)
    #     model_type.check_argument_values(arguments)
    #     cls.check_argument_values(arguments)
    #     logging.info(arguments)
    #     logging.info(extras)
    #
    #     logging.info(f'任务类：{cls}')
    #     logging.info(f'数据加载类：{data_reader_type}')
    #     logging.info(f'模型类：{model_type}')
    #
    #     data_reader: IDataReader = data_reader_type(**arguments)
    #     # todo 扩充特征列信息
    #
    #     model: IModel = model_type(feature_column_dict=data_reader.get_feature_column_dict(), **arguments)
    #
    #     optimizer_type: Type[Optimizer] = get_optimizer(arguments["optimizer"])
    #     optimizer: Optimizer = optimizer_type(lr=arguments["lr"], weight_decay=arguments["l2"])  # noqa
    #
    #     loss: _Loss = get_loss(arguments["loss"])()
    #
    #     return cls(
    #         debug=arguments["debug"],
    #         gpu=arguments["gpu"],
    #         random_seed=arguments["random_seed"],
    #         metrics=arguments["metrics"],
    #         train_mode=arguments["train_mode"],
    #         data_reader=data_reader,
    #         model=model,
    #         epoch=arguments["epoch"],
    #         batch_size=arguments["batch_size"],
    #         optimizer=optimizer,
    #         loss=loss,
    #         callbacks=[],
    #         num_workers=arguments["num_workers"],
    #         dev_freq=arguments["dev_freq"],
    #     )

    def __init__(self,
                 debug: bool,
                 gpu: int,
                 random_seed: int,
                 metrics: List[IMetric],
                 train_mode: TrainMode,
                 data_reader: IDataReader,
                 model: IModel,
                 epoch: int,
                 batch_size: int,
                 optimizer: Optimizer,
                 loss: _Loss,
                 num_workers: int,
                 dev_freq: int,
                 filename: str,
                 monitor: str,
                 monitor_mode: str,
                 patience: int,
                 verbose: int,
                 ):
        self.debug = debug
        if gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu}")
        self.rng = default_rng(random_seed)
        set_torch_seed(random_seed)
        self.metrics: List[IMetric] = metrics
        self.train_mode = train_mode
        self.data_reader = data_reader
        self.train_dataset = TrainDataset(data_reader)
        self.dev_dataset = DevDataset(data_reader)
        self.test_dataset = TestDataset(data_reader)
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.num_workers = num_workers
        self.dev_freq = dev_freq
        self.filename = filename
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.verbose = verbose

    def run(self):
        """执行任务"""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
        )

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{self.filename}.pt"),
            monitor=self.monitor,
            verbose=1,
            save_best_only=True,
            mode=self.monitor_mode,
            save_freq="epoch")

        csv_logger = CSVLogger(
            filename=os.path.join(LOG_DIR, f"{self.filename}.csv"),
            separator="\t")

        early_stopping = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            verbose=1,
            mode=self.monitor_mode)

        history = self.model.fit(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            epochs=self.epoch,
            dev_dataset=self.dev_dataset,
            verbose=self.verbose,
            callbacks=[early_stopping] if self.debug else [model_checkpoint, csv_logger, early_stopping],
            shuffle=True,
            workers=self.num_workers,
            drop_last=True,
            dev_freq=self.dev_freq,
            train_mode=self.train_mode,
        )

        best_epoch, best_dev_logs = history.get_best_epoch_logs(
            monitor=self.monitor,
            monitor_mode=self.monitor_mode,
        )

        self.model.load_best_weights()

        test_csv_logger = CSVLogger(
            filename=os.path.join(LOG_DIR, f"{self.filename}.test.csv"),
            separator="\t")

        test_logs = self.model.evaluate(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[] if self.debug else [test_csv_logger],
            workers=self.num_workers,
        )

        return best_epoch, best_dev_logs, test_logs
