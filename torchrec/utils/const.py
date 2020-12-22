"""全局常量"""
import os
import socket

hostname = socket.gethostname()
print(hostname)

# 工作路径常量
if hostname == 'yanshihongdeMacBook-Pro.local':
    WORK_DIR = '/Users/yansh/Work'  # 工作目录(MacBook Pro)
elif hostname == 'thuir103':
    WORK_DIR = '/work/yansh15'  # 工作目录(cpu103)
else:
    WORK_DIR = '/work/yansh'  # 工作目录(deep/deeper/deepest)

# 数据路径常量
RAW_DATA_DIR = os.path.join(WORK_DIR, 'RawData')  # 原始数据目录
DATASET_DIR = os.path.join(WORK_DIR, 'Dataset')  # 过滤并格式化数据目录(通常包含base_interaction.csv, interaction.csv)
LOG_DIR = os.path.join(WORK_DIR, 'Log')
MODEL_DIR = os.path.join(WORK_DIR, 'Model')
GRID_SEARCH_DIR = os.path.join(WORK_DIR, 'GridSearch')
REPEAT_TASK_DIR = os.path.join(WORK_DIR, 'RepeatTask')

BASE_INTERACTION_CSV = 'base_interaction.csv'  # 用户ID、物品ID、评分、标签、时间戳信息
BASE_INTERACTION_FEATHER = 'base_interaction.feather'  # 用户ID、物品ID、评分、标签、时间戳信息
INTERACTION_CSV = 'interaction.csv'  # 用户ID、物品ID、评分、标签、时间戳、上下文/用户/物品特征（可选）信息
INTERACTION_FEATHER = 'interaction.feather'  # 用户ID、物品ID、评分、标签、时间戳、上下文/用户/物品特征（可选）信息
ITEM_CSV = 'item.csv'  # 物品ID、物品特征（可选）信息
ITEM_FEATHER = 'item.feather'  # 物品ID、物品特征（可选）信息
USER_CSV = 'user.csv'  # 用户ID、用户特征（可选）信息
USER_FEATHER = 'user.feather'  # 用户ID、用户特征（可选）信息
DESCRIPTION_TXT = "description.txt"  # 数据集统计信息
DESCRIPTION_PKL = "description.pkl"  # 数据集统计信息

SPLIT_INDEX_DIR = "SPLIT_INDEX"

SEQUENTIAL_SPLIT_NAME_TEMPLATE = "seq_split_%d_%.2f"  # %d：暖用户要求，%.2f：验证/测试集比例
LEAVE_K_OUT_SPLIT_NAME_TEMPLATE = "leave_k_out_%d_%d"  # %d：暖用户要求，%d：留出个数

TRAIN_INDEX_CSV_TEMPLATE = "%s.train_index.csv"  # 训练集对应的interaction_df的index，%s：划分名称
TRAIN_INDEX_NPY_TEMPLATE = "%s.train_index.npy"  # 训练集对应的interaction_df的index，%s：划分名称
DEV_INDEX_CSV_TEMPLATE = "%s.dev_index.csv"  # 验证集集对应的interaction_df的index，%s：划分名称
DEV_INDEX_NPY_TEMPLATE = "%s.dev_index.npy"  # 验证集集对应的interaction_df的index，%s：划分名称
TEST_INDEX_CSV_TEMPLATE = "%s.test_index.csv"  # 测试集对应的interaction_df的index，%s：划分名称
TEST_INDEX_NPY_TEMPLATE = "%s.test_index.npy"  # 测试集对应的interaction_df的index，%s：划分名称

NEGATIVE_SAMPLE_DIR = "NEGATIVE_SAMPLE"

USER_POS_HIS_SET_DICT_PKL = 'user_pos_his_set_dict.pkl'  # 正向交互信息构成集合按 uid 合并文件
DEV_NEG_CSV_TEMPLATE = 'dev_neg_%d_%d.csv'  # 验证集负采样文件
DEV_NEG_NPY_TEMPLATE = 'dev_neg_%d_%d.npy'  # 验证集负采样文件
TEST_NEG_CSV_TEMPLATE = 'test_neg_%d_%d.csv'  # 测试集负采样文件
TEST_NEG_NPY_TEMPLATE = 'test_neg_%d_%d.npy'  # 测试集负采样文件

HISTORY_DIR = 'HISTORY'

POS_HIS_CSV_TEMPLATE = 'pos_his_%d.csv'  # 正向交互历史信息
POS_HIS_NPY_TEMPLATE = 'pos_his_%d.npy'  # 正向交互历史信息
NEG_HIS_CSV_TEMPLATE = 'neg_his_%d.csv'  # 负向交互历史信息
NEG_HIS_NPY_TEMPLATE = 'neg_his_%d.npy'  # 负向交互历史信息

NEXT_STATE_DIR = 'NEXT_STATE'

POS_NEXT_STATE_CSV_TEMPLATE = 'pos_next_state_%d.csv'  # 正向交互历史信息
POS_NEXT_STATE_NPY_TEMPLATE = 'pos_next_state_%d.npy'  # 正向交互历史信息
NEG_NEXT_STATE_CSV_TEMPLATE = 'neg_next_state_%d.csv'  # 负向交互历史信息
NEG_NEXT_STATE_NPY_TEMPLATE = 'neg_next_state_%d.npy'  # 负向交互历史信息

RL_SAMPLE_DIR = 'RL_SAMPLE'

RL_SAMPLE_CSV_TEMPLATE = 'rl_sample_%d.csv'  # Value-based RL 采样信息（训练集）
RL_SAMPLE_NPY_TEMPLATE = 'rl_sample_%d.npy'  # Value-based RL 采样信息（训练集）

# 数据格式相关常量
SEP = '\t'
SEQ_SEP = ","

# 数据列字符串常量
INDEX = "index"
UID = 'uid'
IID = 'iid'
RATE = "rate"
LABEL = 'label'
TIME = 'time'
IIDS = 'iids'
POS_HIS_LEN = "pos_his_len"
POS_HIS = "pos_his"
NEG_HIS_LEN = "neg_his_len"
NEG_HIS = "neg_his"
POS_STATE_LEN = POS_HIS_LEN
POS_STATE = POS_HIS
NEG_STATE_LEN = NEG_HIS_LEN
NEG_STATE = NEG_HIS
POS_NEXT_STATE_LEN = "pos_next_state_len"
POS_NEXT_STATE = "pos_next_state"
NEG_NEXT_STATE_LEN = "neg_next_state_len"
NEG_NEXT_STATE = "neg_next_state"
RL_SAMPLE = 'rl_sample'  # Value-based RL模型候选集

# 数据集描述词典常量
# 0.数据集描述
INFO = "info"

# 1.基本数据列/特征列相关
BASE_FEATURES = "base_features"
CONTEXT_FEATURES = "context_features"
USER_FEATURES = "user_features"
ITEM_FEATURES = "item_features"

FEATURE_NAME = "feature_name"
FEATURE_TYPE = "feature_type"

# 特征列类型
NUMERIC_COLUMN = "numeric"
CATEGORICAL_COLUMN = "categorical"
NUMERIC_LIST_COLUMN = "numeric_list"
CATEGORICAL_LIST_COLUMN = "categorical_list"

# 特征列描述常量
BUCKET_BOUNDARIES = "bucket_boundaries"
BUCKET_LOG_BASE = "bucket_log_base"
INT_MAP = "int_map"

# 2.用户交互相关
USER_INTERACTION = "user_interaction"

POSITIVE = "positive"
NEGATIVE = "negative"
ALL = "all"
MIN = "min"
MAX = "max"
MEAN = "mean"
MEDIAN = "median"
STD = "std"
