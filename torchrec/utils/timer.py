"""时间统计"""
from time import time


class Timer:
    """时间统计上下文管理"""

    def __init__(self, divided_by: int = 1):
        self.time = None
        self.divided_by = divided_by

    def __enter__(self):
        self.time = time()
        # return self.time

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('时间：%.3f' % ((time() - self.time) / self.divided_by))
