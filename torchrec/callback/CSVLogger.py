"""
CSV记录器
"""
import collections
import csv
from typing import Optional, Dict

import numpy as np

from torchrec.callback.ICallback import ICallback


class CSVLogger(ICallback):
    """CSV记录器"""

    def __init__(self, filename, separator=','):
        self.sep = separator
        self.filename = filename
        self.writer = None
        self.keys = None
        self.csv_file = None
        super().__init__()

    def on_train_begin(self, logs: Optional[Dict] = None):
        self.csv_file = open(self.filename, "w")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (isinstance(k, list) or isinstance(k, tuple) or isinstance(k, np.ndarray)) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return str(k)

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs: Optional[Dict] = None):
        self.csv_file.close()
        self.writer = None
