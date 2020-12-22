"""
任务类相关函数
"""
from typing import Dict, Type

from torchrec.task.ITask import ITask
from torchrec.task.Task import Task

_task_classes: Dict[str, Type[ITask]] = {
    "normal": Task,
}

task_name_list = _task_classes.keys()


def get_task_type(task_name: str) -> Type[ITask]:
    """根据任务类型返回任务类"""
    if (not isinstance(task_name, str)) or (task_name not in _task_classes):
        raise ValueError(f"task_name参数不合法: {task_name}")
    return _task_classes[task_name]
