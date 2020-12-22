"""
系统相关模块
"""
import logging

from torchrec.utils.const import *


def init_console_logger(level=logging.INFO) -> None:
    """
    初始化终端日志打印
    """
    logging.basicConfig(level=level, format='%(asctime)s %(filename)s %(funcName)s :: %(message)s')


def check_dir_and_mkdir(path: str) -> None:
    """
    检查路径是否存在，如果不存在则创建路劲，如果输入是文件名则自动提取路径。
    """
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        logging.info(f'新建文件夹：{dirname}')
        os.makedirs(dirname)
    return


def check_important_dirs_and_mkdir():
    """
    检查模块重要文件夹是否已经建立
    """
    if not os.path.exists(DATASET_DIR):
        raise RuntimeError(f"数据集目录不存在：{DATASET_DIR}")
    # check_dir_and_mkdir(LOG_AND_RESULT_DIR)
    # check_dir_and_mkdir(LOG_DIR)
    # check_dir_and_mkdir(RESULT_DIR)
    # check_dir_and_mkdir(MODEL_DIR)
    # check_dir_and_mkdir(GRID_SEARCH_DIR)
    # check_dir_and_mkdir(COMMAND_DIR)
