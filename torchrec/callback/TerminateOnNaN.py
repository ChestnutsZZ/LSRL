"""
当损失为异常值时终止训练
"""
from typing import Optional, Dict

import numpy as np

from torchrec.callback.ICallback import ICallback


class TerminateOnNaN(ICallback):
    """当损失为异常值时终止训练"""

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'Batch {batch}: Invalid loss, terminating training')
                self.model.stop_training = True
