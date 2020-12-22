"""
进度条
"""
import copy
import os
import sys
import time
from typing import Optional, List, Tuple, Any, Dict

import numpy as np

from torchrec.callback.ICallback import ICallback


class Progbar(object):
    """进度条"""

    def __init__(self,
                 target: Optional[int] = None,
                 width: int = 30,
                 verbose: int = 1,
                 interval: float = 0.05):
        """
        :param target:              总数
        :param width:               进度条宽度
        :param verbose:             0：不显示，1：显示进度+轮次结果，2：显示轮次结果
        :param interval:            最短更新时间间隔
        """
        self.target: Optional[int] = target  # 总数
        self.width: int = width  # 进度条宽度
        self.verbose: int = verbose  # 显示粒度
        self.interval: float = interval  # 最短更新时间间隔

        # 能否交互性的显示
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules
                                 or 'PYCHARM_HOSTED' in os.environ)

        self._seen_so_far = 0  # 当前数量

        self._values = {}  # 记录键值对
        self._values_order = []  # 记录键顺序

        self._start = time.time()  # 开始时间
        self._last_update = 0  # 最后一次更新时间

    def update(self,
               current: int,
               values: Optional[List[Tuple[str, Any]]] = None,
               finalize: Optional[bool] = None
               ) -> None:
        """
        更新进度条
        :param current:  当前数量
        :param values:   键值对列表
        :param finalize: 是否为最终状态（如果存在target，可以通过target判断，如果判断不出，视为False）
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            info = '\r' if self._dynamic_display else '\n'

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                info += ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    info += ('=' * (prog_width - 1))
                    info += '>' if current < self.target else '='
                info += ('.' * (self.width - prog_width))
                info += ']'
            else:
                info += '%7d/Unknown' % current

            info += ' - %.0fs' % (now - self._start)

            time_per_unit = (now - self._start) / current if current != 0 else 0

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, 'batch')
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, 'batch')
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, 'batch')
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info += ' - ETA: %s' % eta_format

            for k in self._values_order:
                info += ' - %s:' % k
                info += ' %s' % self._values[k]

            if finalize:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if finalize:
                info = ''
                if self.target:
                    numdigits = int(np.log10(self.target)) + 1
                    count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                    info += count
                else:
                    info += '%7d/Unknown' % current
                info += ' - %.0fs' % (now - self._start)
                for k in self._values_order:
                    info += ' - %s:' % k
                    if self._values[k] > 1e-3:
                        info += ' %.4f' % self._values[k]
                    else:
                        info += ' %.4e' % self._values[k]
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now


class ProgbarLogger(ICallback):
    """进度记录器，显示进度、loss与其他评价指标"""

    def __init__(self):
        super().__init__()
        self.seen: int = 0
        self.target: Optional[int] = None
        self.verbose: int = 1
        self.epochs: int = 1
        self._called_in_fit: bool = False
        self.progbar: Optional[Progbar] = None

    def set_params(self, params: Dict):
        """设置参数"""
        self.verbose = params['verbose']
        self.epochs = params['epochs']
        self.target = params['batches']
        self.progbar = Progbar(
            target=self.target,
            verbose=self.verbose)

    def _reset_progbar(self):
        """重置"""
        self.seen: int = 0
        self.progbar: Optional[Progbar] = Progbar(
            target=self.target,
            verbose=self.verbose)

    def _batch_update_progbar(self, logs: Optional[Dict] = None):
        logs = copy.copy(logs) if logs else {}
        logs.pop('batch', None)
        add_seen = 1
        self.seen += add_seen
        self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _finalize_progbar(self, logs: Dict):
        """结束"""
        if self.target is None:
            self.target = self.seen
            self.progbar.target = self.seen
        logs = logs or {}
        self.progbar.update(self.seen, list(logs.items()), finalize=True)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始的时候"""
        self._called_in_fit = True  # 确保验证时不使用

    def on_test_begin(self, logs: Optional[Dict] = None):
        """测试开始的时候，验证时不使用"""
        if not self._called_in_fit:
            self._reset_progbar()

    def on_predict_begin(self, logs: Optional[Dict] = None):
        """预测开始的时候"""
        self._reset_progbar()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """轮次开始时调用"""
        self._reset_progbar()
        if self.verbose != 0 and self.epochs > 1:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """训练批次结束后调用"""
        self._batch_update_progbar(logs)

    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次结束的时候"""
        if not self._called_in_fit:
            self._batch_update_progbar(logs)

    def on_predict_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """预测批次结束的时候"""
        self._batch_update_progbar()  # Don't pass prediction results.

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束时调用"""
        self._finalize_progbar(logs)

    def on_test_end(self, logs: Optional[Dict] = None):
        """验证/测试结束的时候"""
        if not self._called_in_fit:
            self._finalize_progbar(logs)

    def on_predict_end(self, logs: Optional[Dict] = None):
        """预测结束的时候"""
        self._finalize_progbar(logs)
