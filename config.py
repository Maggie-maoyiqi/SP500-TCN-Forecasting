# ===== 路径和参数配置 =====

TRAIN_PATH  = 'dataset/3010train.csv'
TEST_PATH   = 'dataset/3010test.csv'
DGS10_PATH  = 'dataset/DGS10.csv'
GDPC1_PATH  = 'dataset/GDPC1.csv'

SEED          = 42
LOOKBACK      = 60   # 使用过去60天的数据
N_FIRST_STAGE = 200  # 第一阶段训练次数
N_TOP_MODELS  = 5    # 选出最佳模型数

RESULTS_DIR = 'results'
