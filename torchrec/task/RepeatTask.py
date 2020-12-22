import pandas as pd
from numpy.random._generator import default_rng  # noqa
from pandas import DataFrame
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, List, Type

from torchrec.data.IDataReader import IDataReader
from torchrec.metric import IMetric
from torchrec.model import IModel
from torchrec.task import TrainMode
from torchrec.task.ITask import ITask
from torchrec.task.Task import Task
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *


class RepeatTask(ITask):
    def __init__(self,
                 repeat_num: int,
                 gpu: int,
                 random_seed: int,
                 metrics: List[IMetric],
                 train_mode: TrainMode,
                 data_reader_type: Type[IDataReader],
                 data_reader_params: Dict,
                 model_type: Type[IModel],
                 model_params: Dict,
                 epoch: int,
                 batch_size: int,
                 optimizer_type: Type[Optimizer],
                 optimizer_params: Dict,
                 loss: _Loss,
                 num_workers: int,
                 dev_freq: int,
                 monitor: str,
                 monitor_mode: str,
                 patience: int,
                 verbose: int,
                 ):
        self.repeat_num = repeat_num
        self.gpu = gpu
        self.random_seed = random_seed
        self.metrics: List[IMetric] = metrics
        self.train_mode = train_mode
        self.data_reader_type = data_reader_type
        self.data_reader_params = data_reader_params
        self.model_type = model_type
        self.model_params = model_params
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.num_workers = num_workers
        self.dev_freq = dev_freq
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.verbose = verbose

        self.log_filename = os.path.join(
            REPEAT_TASK_DIR,
            f"{self.model_type.__name__}"
            f"_{self.data_reader_params['dataset']}"
            f"_{self.loss.__class__.__name__}"
            f"_repeat_{self.repeat_num}.csv")

    def run(self):
        task_logs: Dict[str, List] = {}
        for repeat_index in range(self.repeat_num):
            random_seed = self.random_seed + repeat_index
            assert "random_seed" in self.data_reader_params
            self.data_reader_params["random_seed"] = random_seed
            assert "random_seed" in self.model_params
            self.model_params["random_seed"] = random_seed
            params: Dict[str, str] = {
                "model": self.model_type.__name__,
                "dataset": self.data_reader_params["dataset"],
                "seed": str(random_seed),
                "train": self.train_mode.value,
                "epoch": str(self.epoch),
                "batch_size": str(self.batch_size),
                "optimizer": str(self.optimizer_type.__name__),
                "loss": self.loss.__class__.__name__,
                "dev_freq": str(self.dev_freq),
                "monitor": self.monitor,
                "patience": str(self.patience),
            }
            params.update(self.data_reader_params)
            params.update(self.model_params)
            params.update(self.optimizer_params)
            filename: str = self.log_filename.replace(".csv", f".{repeat_index}.csv")
            data_reader = self.data_reader_type(**self.data_reader_params)
            model = self.model_type(**self.model_params)
            optimizer = self.optimizer_type(params=model.get_parameters(), **self.optimizer_params)  # noqa
            best_epoch, best_dev_logs, test_logs = Task(
                debug=False,
                gpu=self.gpu,
                random_seed=self.random_seed,
                metrics=self.metrics,
                train_mode=self.train_mode,
                data_reader=data_reader,
                model=model,
                epoch=self.epoch,
                batch_size=self.batch_size,
                optimizer=optimizer,
                loss=self.loss,
                num_workers=self.num_workers,
                dev_freq=self.dev_freq,
                filename=filename,
                monitor=self.monitor,
                monitor_mode=self.monitor_mode,
                patience=self.patience,
                verbose=self.verbose,
            ).run()
            params["best_epoch"] = best_epoch
            params.update({f"dev_{key}": best_dev_logs[key] for key in best_dev_logs})
            params.update({f"test_{key}": test_logs[key] for key in test_logs})
            for key in params:
                task_logs.setdefault(key, []).append(params[key])
            repeat_task_df: DataFrame = pd.DataFrame(task_logs)
            repeat_task_df.to_csv(
                path_or_buf=self.log_filename,
                sep="\t",
            )

    @classmethod
    def create_from_console(cls):
        pass

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass
