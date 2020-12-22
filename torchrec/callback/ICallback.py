"""
回调函数接口类
"""
from abc import ABC
from typing import Optional, Dict, Callable


def empty_implementation(method: Callable):
    """标明一个函数的实现没有任何语句，可以忽略，通常为基类函数"""
    method._empty_implementation = True
    return method


def is_empty_implementation(method: Callable):
    """检查函数是否为空实现"""
    return getattr(method, '_empty_implementation', False)


class ICallback(ABC):
    """回调函数接口类"""

    def __init__(self):
        self.model = None
        self.params: Optional[Dict] = None

    def set_params(self, params):
        """设置参数"""
        self.params = params

    def set_model(self, model):
        """引用推荐模型"""
        self.model = model

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """轮次开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束时调用"""
        pass

    @empty_implementation
    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """训练批次开始时调用"""
        pass

    @empty_implementation
    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """训练批次结束后调用"""
        pass

    @empty_implementation
    def on_test_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次开始的时候"""
        pass

    @empty_implementation
    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次结束的时候"""
        pass

    @empty_implementation
    def on_predict_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """预测批次开始的时候"""
        pass

    @empty_implementation
    def on_predict_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """预测批次结束的时候"""
        pass

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始的时候"""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束的时候"""
        pass

    def on_test_begin(self, logs: Optional[Dict] = None):
        """验证/测试开始的时候"""
        pass

    def on_test_end(self, logs: Optional[Dict] = None):
        """验证/测试结束的时候"""
        pass

    def on_predict_begin(self, logs: Optional[Dict] = None):
        """预测开始的时候"""
        pass

    def on_predict_end(self, logs: Optional[Dict] = None):
        """预测结束的时候"""
        pass

    def implements_train_batch_hooks(self):
        """检查是否在训练批次上执行功能"""
        return (not is_empty_implementation(self.on_train_batch_begin) or
                not is_empty_implementation(self.on_train_batch_end))

    def implements_test_batch_hooks(self):
        """检查是否在测试批次上执行功能"""
        return (not is_empty_implementation(self.on_test_batch_begin) or
                not is_empty_implementation(self.on_test_batch_end))

    def implements_predict_batch_hooks(self):
        """检查是否在预测批次上执行功能"""
        return (not is_empty_implementation(self.on_predict_batch_begin) or
                not is_empty_implementation(self.on_predict_batch_end))
