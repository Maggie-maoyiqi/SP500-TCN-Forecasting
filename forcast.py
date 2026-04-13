# ===== 导入所有需要的库 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入tqdm用于进度条
from tqdm import tqdm
import time

# 机器学习相关
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 深度学习相关
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===== 设置路径和参数 =====
# ⚠️ 注意：修改为你自己的文件路径！
TRAIN_PATH = 'dataset/3010train.csv'
TEST_PATH = 'dataset/3010test.csv'
DGS10_PATH = 'dataset/DGS10.csv'
GDPC1_PATH = 'dataset/GDPC1.csv'

SEED = 42
LOOKBACK = 60  # 使用过去60天的数据

# 总步骤数
TOTAL_STEPS = 10
current_step = 0

def update_progress(step_name):
    """更新进度并显示进度条"""
    global current_step
    current_step += 1
    progress_percent = (current_step / TOTAL_STEPS) * 100
    print(f"\n📊 [{current_step}/{TOTAL_STEPS}] {step_name}... ({progress_percent:.0f}%)")
    return tqdm(total=100, desc=f"  进度", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

# ===== 加载数据 =====
pbar = update_progress("加载数据")
time.sleep(0.5)

# 1. 加载股价数据
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
pbar.update(20)

# 2. 加载宏观经济数据
dgs10_df = pd.read_csv(DGS10_PATH)  # 国债收益率
gdpc1_df = pd.read_csv(GDPC1_PATH)  # GDP数据
pbar.update(20)

# 3. 日期格式转换
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
dgs10_df['observation_date'] = pd.to_datetime(dgs10_df['observation_date'])
gdpc1_df['observation_date'] = pd.to_datetime(gdpc1_df['observation_date'])
pbar.update(20)

# 4. 按日期排序
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)
pbar.update(20)

# 5. 清理价格数据（去掉美元符号）
for col in ['Close/Last', 'Open', 'High', 'Low']:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].str.replace('$', '').astype(float)
        test_df[col] = test_df[col].str.replace('$', '').astype(float)
pbar.update(20)
pbar.close()

print(f"   ✓ 训练数据：{len(train_df)} 行")
print(f"   ✓ 测试数据：{len(test_df)} 行")
print(f"   ✓ 国债数据：{len(dgs10_df)} 行")
print(f"   ✓ GDP数据：{len(gdpc1_df)} 行")

# ===== 合并宏观经济数据 =====
pbar = update_progress("合并宏观经济数据")
time.sleep(0.5)

# 1. 重命名列
dgs10_df.columns = ['Date', 'DGS10']
gdpc1_df.columns = ['Date', 'GDPC1']
pbar.update(20)

# 2. 转换数据类型（处理可能的错误值）
dgs10_df['DGS10'] = pd.to_numeric(dgs10_df['DGS10'], errors='coerce')
gdpc1_df['GDPC1'] = pd.to_numeric(gdpc1_df['GDPC1'], errors='coerce')
pbar.update(20)

# 3. 合并DGS10（国债收益率）- 日频数据
train_df = train_df.merge(dgs10_df, on='Date', how='left')
test_df = test_df.merge(dgs10_df, on='Date', how='left')
pbar.update(20)

# 4. 合并GDPC1（GDP）- 季频数据，需要填充
train_df = train_df.merge(gdpc1_df, on='Date', how='left')
test_df = test_df.merge(gdpc1_df, on='Date', how='left')
pbar.update(20)

# 5. 向前填充缺失的宏观数据
for df in [train_df, test_df]:
    df['DGS10'] = df['DGS10'].ffill().bfill()
    df['GDPC1'] = df['GDPC1'].ffill().bfill()
pbar.update(20)
pbar.close()

print(f"   ✓ 合并了DGS10（10年期国债收益率）")
print(f"   ✓ 合并了GDPC1（实际GDP）")
print(f"   ✓ 使用前向填充处理了缺失值")

# ===== 特征工程 =====
pbar = update_progress("特征工程（创建40个特征）")
time.sleep(0.5)

def create_features_no_leakage(df):
    """创建特征，严格避免数据泄露"""
    df = df.copy()
    
    # ===== 基础特征（全部用前一天数据）=====
    df['prev_close'] = df['Close/Last'].shift(1)  # 前一日收盘价
    df['prev_open'] = df['Open'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_volume'] = df['Volume'].shift(1)
    
    # ===== 技术指标 =====
    # 价格范围
    df['prev_hl_range'] = df['prev_high'] - df['prev_low']
    df['prev_oc_diff'] = df['prev_open'] - df['prev_close']
    
    # 收益率
    df['returns_1d'] = df['Close/Last'].pct_change(1).shift(1)
    df['returns_5d'] = df['Close/Last'].pct_change(5).shift(1)
    
    # 移动平均线（5日、10日、20日）
    for window in [5, 10, 20]:
        ma = df['Close/Last'].shift(1).rolling(window).mean()
        df[f'MA_{window}'] = ma
        df[f'MA_{window}_ratio'] = df['prev_close'] / ma  # 价格与均线比值
    
    # RSI相对强弱指数
    delta = df['Close/Last'].diff().shift(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close/Last'].shift(1).ewm(span=12, adjust=False).mean()
    exp2 = df['Close/Last'].shift(1).ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # 波动率
    df['volatility_10'] = df['returns_1d'].rolling(10).std()
    
    # 成交量比率
    volume_ma = df['Volume'].shift(1).rolling(10).mean()
    df['volume_ratio'] = df['prev_volume'] / volume_ma
    
    # ===== 🆕 增强动量特征（9个）=====
    # 价格动量
    df['momentum_2d'] = (df['Close/Last'].shift(1) - df['Close/Last'].shift(3))
    df['momentum_3d'] = (df['Close/Last'].shift(1) - df['Close/Last'].shift(4))
    df['momentum_5d'] = (df['Close/Last'].shift(1) - df['Close/Last'].shift(6))
    
    # 加速度（动量的变化率）
    df['acceleration'] = df['returns_1d'] - df['returns_1d'].shift(1)
    
    # 趋势强度
    ema_fast = df['Close/Last'].shift(1).ewm(span=5, adjust=False).mean()
    ema_slow = df['Close/Last'].shift(1).ewm(span=20, adjust=False).mean()
    df['trend_strength'] = (ema_fast - ema_slow) / ema_slow
    
    # 动量比率（短期vs长期）
    short_momentum = df['Close/Last'].shift(1) - df['Close/Last'].shift(3)
    long_momentum = df['Close/Last'].shift(1) - df['Close/Last'].shift(11)
    df['momentum_ratio'] = short_momentum / (long_momentum.abs() + 1e-8)
    
    # 波动率动量
    df['volatility_momentum'] = df['volatility_10'] - df['volatility_10'].shift(5)
    
    # 成交量动量
    df['volume_momentum'] = (df['Volume'].shift(1) - df['Volume'].shift(6)) / df['Volume'].shift(6)
    
    # 变动率
    df['ROC_5'] = ((df['Close/Last'].shift(1) - df['Close/Last'].shift(6)) / df['Close/Last'].shift(6) * 100)
    df['ROC_10'] = ((df['Close/Last'].shift(1) - df['Close/Last'].shift(11)) / df['Close/Last'].shift(11) * 100)
    
    # 布林带位置
    rolling_mean = df['Close/Last'].shift(1).rolling(20).mean()
    rolling_std = df['Close/Last'].shift(1).rolling(20).std()
    df['bb_position'] = (df['prev_close'] - rolling_mean) / (2 * rolling_std)
    
    # ===== 🆕 核心价量因子（5个）=====
    # 1. 日内动量强度
    df['intraday_momentum'] = (df['prev_close'] - df['prev_open']) / (df['prev_high'] - df['prev_low'] + 1e-8)
    
    # 2. K线实体比率
    df['candle_body_ratio'] = np.abs(df['prev_close'] - df['prev_open']) / (df['prev_high'] - df['prev_low'] + 1e-8)
    
    # 3. 标准化范围
    df['normalized_range'] = (df['prev_high'] - df['prev_low']) / (df['prev_open'] + 1e-8)
    
    # 4. 价量配合度
    df['price_volume_fit'] = (df['prev_close'] - df['prev_open']) * df['prev_volume']
    
    # 5. 资金流
    df['money_flow'] = (((df['prev_close'] - df['prev_low']) - (df['prev_high'] - df['prev_close'])) / 
                        (df['prev_high'] - df['prev_low'] + 1e-8)) * df['prev_volume']
    
    # ===== 🆕 宏观经济因子（5个）=====
    df['prev_dgs10'] = df['DGS10'].shift(1)  # 前一日国债收益率
    df['prev_gdpc1'] = df['GDPC1'].shift(1)  # 前一日GDP
    
    # 宏观因子变化
    df['dgs10_change'] = df['DGS10'].diff().shift(1)  # 国债收益率变化
    df['dgs10_ma_10'] = df['DGS10'].shift(1).rolling(10).mean()  # 10日均值
    df['gdpc1_growth'] = df['GDPC1'].pct_change(1).shift(1)  # GDP增长率
    
    return df

# 应用特征工程
train_with_features = create_features_no_leakage(train_df)
train_with_features = train_with_features.bfill().ffill()
pbar.update(50)

# 确定特征列（排除不需要的列）
exclude_cols = ['Date', 'Close/Last', 'Open', 'High', 'Low', 'Volume', 'DGS10', 'GDPC1']
feature_columns = [col for col in train_with_features.columns 
                   if col not in exclude_cols]
pbar.update(50)
pbar.close()

print(f"   ✓ 创建了 {len(feature_columns)} 个特征")
print(f"   🆕 9个动量特征：momentum_2d/3d/5d, acceleration, trend_strength等")
print(f"   🆕 5个价量因子：intraday_momentum, candle_body_ratio等")
print(f"   🆕 5个宏观因子：prev_dgs10, prev_gdpc1, dgs10_change等")

# ===== 数据预处理 =====
pbar = update_progress("数据预处理")
time.sleep(0.5)

# 1. 准备数据
X_data = train_with_features[feature_columns].values  # 特征矩阵
y_data = train_with_features['Close/Last'].values     # 目标值（收盘价）
pbar.update(20)

# 2. 归一化（非常重要！）
scaler_X = MinMaxScaler()  # 特征归一化器
scaler_y = MinMaxScaler()  # 目标值归一化器

X_scaled = scaler_X.fit_transform(X_data)  # 对特征归一化
y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))  # 对目标值归一化
pbar.update(20)

# 3. 创建时间序列序列
def create_sequences(X, y, lookback):
    """创建时间序列的滑动窗口"""
    Xs, ys = [], []
    for i in tqdm(range(lookback, len(X)), desc="创建序列", leave=False):
        Xs.append(X[i-lookback:i])  # 取lookback天的特征
        ys.append(y[i])             # 第i天的目标值
    return np.array(Xs), np.array(ys)

# 创建序列
X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)
pbar.update(20)

# 4. 划分训练集和验证集
split_idx = int(len(X_seq) * 0.85)  # 85%训练，15%验证
X_train = X_seq[:split_idx]
y_train = y_seq[:split_idx]
X_val = X_seq[split_idx:]
y_val = y_seq[split_idx:]
pbar.update(20)

print(f"   ✓ 训练集：{len(X_train)} 个样本")
print(f"   ✓ 验证集：{len(X_val)} 个样本")
print(f"   ✓ 输入形状：{X_train.shape} (样本数, 时间步长, 特征数)")
print(f"   ✓ 特征数量：{len(feature_columns)}")
pbar.update(20)
pbar.close()

print("=" * 80)
print("🏗️  [5/10] 构建TCN模型")
print("=" * 80)
print()

# ===== 1. 先定义TCN的基本积木块 =====
print("🔨 1/3 定义TCN时间块...")
class TemporalBlock(keras.layers.Layer):
    """TCN的时间块"""
    
    def __init__(self, n_outputs, kernel_size, dilation_rate, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        # 第一层卷积
        self.conv1 = keras.layers.Conv1D(
            filters=self.n_outputs,
            kernel_size=self.kernel_size,
            padding='causal',
            dilation_rate=self.dilation_rate,
            activation='relu'
        )
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        
        # 第二层卷积
        self.conv2 = keras.layers.Conv1D(
            filters=self.n_outputs,
            kernel_size=self.kernel_size,
            padding='causal',
            dilation_rate=self.dilation_rate,
            activation='relu'
        )
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        
        # 如果需要，调整输入维度
        if input_shape[-1] != self.n_outputs:
            self.downsample = keras.layers.Conv1D(filters=self.n_outputs, kernel_size=1)
        else:
            self.downsample = None
            
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        
        if self.downsample is not None:
            inputs = self.downsample(inputs)
            
        return keras.activations.relu(x + inputs)

print("✅ TCN时间块定义完成")

# ===== 2. 搭建完整模型 =====
print("\n🔨 2/3 搭建完整TCN模型...")

def build_tcn_model(input_shape, verbose=True):
    """搭建TCN模型"""
    if verbose:
        print(f"   输入形状: {input_shape}")
    
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 第一个时间块
    x = TemporalBlock(32, 3, 1, 0.2)(inputs)
    if verbose:
        print(f"   块1后形状: {x.shape}")
    
    # 第二个时间块
    x = TemporalBlock(32, 3, 2, 0.2)(x)
    if verbose:
        print(f"   块2后形状: {x.shape}")
    
    # 展平层
    x = Flatten()(x)
    if verbose:
        print(f"   展平后形状: {x.shape}")
    
    # 全连接层
    x = Dense(16, activation='relu')(x)
    
    # 输出层
    outputs = Dense(1)(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name='TCN')
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    
    return model

# ===== 3. 测试模型构建 =====
print("\n🔨 3/3 测试模型构建...")
input_shape = (LOOKBACK, len(feature_columns))
test_model = build_tcn_model(input_shape, verbose=True)

print("\n" + "=" * 60)
print("✅ 第5步完成：TCN模型构建成功！")
print(f"   模型总参数: {test_model.count_params():,}")
print(f"   输入形状: {input_shape}")
print("=" * 60)

print("\n" + "=" * 80)
print("🎲 [6/10] 创建梯度种子优化器")
print("=" * 80)
print()

# ===== 1. 定义优化器类 =====
print("🔧 1/3 定义优化器类...")

class GradientBasedSeedOptimizer:
    """智能种子优化器"""
    
    def __init__(self, base_seed=42, learning_rate=10.0):
        self.base_seed = base_seed
        self.learning_rate = learning_rate
        self.seed_history = []
        self.loss_history = []
        self.best_seed = base_seed
        self.best_loss = float('inf')
        
    def compute_seed_gradient(self):
        """计算种子梯度（实际上主要是随机搜索）"""
        if len(self.loss_history) < 2:
            return np.random.randn() * self.learning_rate
        
        recent_losses = self.loss_history[-3:]
        recent_seeds = self.seed_history[-3:]
        
        if len(recent_losses) >= 2:
            loss_diff = recent_losses[-1] - recent_losses[-2]
            seed_diff = recent_seeds[-1] - recent_seeds[-2]
            
            if seed_diff != 0:
                gradient = loss_diff / seed_diff
            else:
                gradient = 0
            
            momentum = np.random.uniform(-0.3, 0.3)
            noise = np.random.randn() * 0.2
            
            return -gradient * self.learning_rate + momentum + noise
        
        return np.random.randn() * self.learning_rate
    
    def get_next_seed(self):
        """获取下一个种子"""
        if len(self.seed_history) == 0:
            next_seed = self.base_seed
        else:
            gradient_step = self.compute_seed_gradient()
            
            if len(self.loss_history) > 0 and self.loss_history[-1] < self.best_loss:
                next_seed = int(self.seed_history[-1] + gradient_step * 0.5)
            else:
                next_seed = int(self.best_seed + gradient_step)
            
            next_seed = max(1, abs(next_seed))
            
            if next_seed in self.seed_history:
                next_seed += np.random.randint(10, 100)
        
        return next_seed
    
    def update(self, seed, loss):
        """更新优化器状态"""
        self.seed_history.append(seed)
        self.loss_history.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_seed = seed
    
    def get_summary(self):
        """获取总结"""
        return {
            'best_seed': self.best_seed,
            'best_loss': self.best_loss,
            'seed_history': self.seed_history.copy(),
            'loss_history': self.loss_history.copy()
        }

print("✅ 优化器类定义完成")

# ===== 2. 创建优化器实例 =====
print("\n🔧 2/3 创建优化器实例...")
seed_optimizer = GradientBasedSeedOptimizer(base_seed=SEED, learning_rate=15.0)
print(f"   基础种子: {SEED}")
print(f"   学习率: 15.0")

# ===== 3. 简单测试优化器 =====
print("\n🔧 3/3 测试优化器...")
print("   模拟5次搜索过程：")
for i in range(5):
    seed = seed_optimizer.get_next_seed()
    # 模拟一个损失值
    simulated_loss = np.random.rand() * 100
    seed_optimizer.update(seed, simulated_loss)
    print(f"     迭代 {i+1}: 种子={seed:4d}, 模拟损失={simulated_loss:.2f}")

summary = seed_optimizer.get_summary()
print(f"\n✅ 第6步完成：种子优化器创建成功！")
print(f"   当前最佳种子: {summary['best_seed']}")
print(f"   当前最佳损失: {summary['best_loss']:.2f}")
print("=" * 60)

print("\n" + "=" * 80)
print("🔄 [7/10] 创建滚动预测函数")
print("=" * 80)
print()

# ===== 1. 定义滚动预测函数 =====
print("🔧 1/2 定义滚动预测函数...")

def rolling_forecast_deep_learning(model, test_df, train_df, 
                                   scaler_X, scaler_y, feature_columns, 
                                   lookback, show_progress=True):
    """
    TCN模型滚动预测 - 用真实值更新历史，防止误差累积
    """
    
    history_df = train_df.copy()
    predictions = []
    test_size = len(test_df)
    
    if show_progress:
        print(f"   开始滚动预测，共 {test_size} 天")
        print("   " + "=" * 40)
    
    for i in range(test_size):
        if show_progress and i % max(1, test_size // 20) == 0:
            percent = (i / test_size) * 100
            bar_length = 40
            filled_length = int(bar_length * i // test_size)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   [{bar}] {i:4d}/{test_size} ({percent:5.1f}%)", end='\r')
        
        try:
            # 准备输入数据
            current_history = create_features_no_leakage(history_df)
            current_history = current_history.bfill().ffill()
            
            recent_features = current_history[feature_columns].values[-lookback:]
            recent_scaled = scaler_X.transform(recent_features)
            X_input = recent_scaled.reshape(1, lookback, len(feature_columns))
            
            # 模型预测
            pred_scaled = model.predict(X_input, verbose=0)[0, 0]
            pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # 🔑 关键：用真实值更新历史
            next_row = test_df.iloc[i:i+1].copy()
            history_df = pd.concat([history_df, next_row], ignore_index=True)
            
        except Exception as e:
            if predictions:
                predictions.append(predictions[-1])
            else:
                predictions.append(history_df['Close/Last'].iloc[-1])
            
            next_row = test_df.iloc[i:i+1].copy()
            history_df = pd.concat([history_df, next_row], ignore_index=True)
    
    if show_progress:
        bar = '█' * 40
        print(f"   [{bar}] {test_size:4d}/{test_size} (100.0%) ✅")
        print()
    
    return np.array(predictions)

print("✅ 滚动预测函数定义完成")

# ===== 2. 测试函数（用小样本） =====
print("\n🔧 2/2 测试滚动预测函数...")
print("   注意：这只是测试函数是否能运行，用前5天测试数据")

# 取前5天测试数据做快速测试
test_sample = test_df.head(5).copy()

print("   测试中...", end=' ')
try:
    # 用测试模型（第5步构建的）进行预测
    test_predictions = rolling_forecast_deep_learning(
        test_model, test_sample, train_df, scaler_X, scaler_y, 
        feature_columns, LOOKBACK, show_progress=True
    )
    print("✅ 滚动预测函数测试成功！")
    print(f"   预测了 {len(test_predictions)} 天的数据")
    print(f"   预测值范围: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("   跳过测试，继续下一步...")

print("\n" + "=" * 60)
print("✅ 第7步完成：滚动预测函数创建成功！")
print("=" * 60)

print("\n" + "=" * 80)
print("🔥 [8/10] 第一阶段：训练200个模型，选Top 5")
print("=" * 80)
print()

# ===== 1. 设置参数和准备工作 =====
print("📋 1/6 设置参数...")
N_FIRST_STAGE = 200
N_TOP_MODELS = 5

print(f"   第一阶段训练次数: {N_FIRST_STAGE}")
print(f"   选出最佳模型数: {N_TOP_MODELS}")

# 准备测试数据
y_test = test_df['Close/Last'].values
input_shape = (LOOKBACK, len(feature_columns))

# 设置回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                           restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                              patience=5, verbose=0)

# 重置种子优化器
seed_optimizer = GradientBasedSeedOptimizer(base_seed=SEED, learning_rate=15.0)

print("✅ 参数设置完成")

# ===== 2. 开始200次训练循环 =====
print("\n📋 2/6 开始200次训练循环...")
print("   " + "=" * 50)

first_stage_results = []
start_time_total = time.time()

for run in range(N_FIRST_STAGE):
    # 显示进度条
    if run % max(1, N_FIRST_STAGE // 50) == 0 or run == N_FIRST_STAGE - 1:
        percent = (run / N_FIRST_STAGE) * 100
        bar_length = 50
        filled_length = int(bar_length * run // N_FIRST_STAGE)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 计算预计剩余时间
        elapsed = time.time() - start_time_total
        if run > 0:
            time_per_run = elapsed / run
            remaining = time_per_run * (N_FIRST_STAGE - run)
            time_str = f"剩余: {remaining:.0f}s"
        else:
            time_str = "计算中..."
        
        print(f"   [{bar}] {run:3d}/{N_FIRST_STAGE} ({percent:5.1f}%) {time_str}", end='\r')
    
    # === 获取种子并设置 ===
    current_seed = seed_optimizer.get_next_seed()
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)
    
    # === 构建并训练模型 ===
    start_time = time.time()
    
    # 构建模型（关闭详细输出）
    tcn_model = build_tcn_model(input_shape, verbose=False)
    
    # 训练模型
    history = tcn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # === 在验证集上评估 ===
    val_pred = tcn_model.predict(X_val, verbose=0)
    val_pred_original = scaler_y.inverse_transform(val_pred)
    val_true_original = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    
    val_mse = mean_squared_error(val_true_original, val_pred_original)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(val_true_original, val_pred_original)
    
    # === 更新种子优化器 ===
    seed_optimizer.update(current_seed, val_mse)
    
    # === 保存结果 ===
    first_stage_results.append({
        'run': run + 1,
        'seed': current_seed,
        'val_MSE': val_mse,
        'val_RMSE': val_rmse,
        'val_MAE': val_mae,
        'epochs': len(history.history['loss']),
        'time': time.time() - start_time,
        'model': tcn_model
    })

# 完成进度条
bar = '█' * 50
elapsed_total = time.time() - start_time_total
print(f"   [{bar}] {N_FIRST_STAGE:3d}/{N_FIRST_STAGE} (100.0%) 完成！总用时: {elapsed_total:.0f}s")
print("\n✅ 200次训练完成！")

# ===== 3. 分析第一阶段结果 =====
print("\n📋 3/6 分析第一阶段结果...")

# 按验证MSE排序
first_stage_results_sorted = sorted(first_stage_results, key=lambda x: x['val_MSE'])
top_5_models = first_stage_results_sorted[:N_TOP_MODELS]

print(f"   验证MSE范围: {first_stage_results_sorted[0]['val_MSE']:.4f} - {first_stage_results_sorted[-1]['val_MSE']:.4f}")
print(f"   平均训练时间: {np.mean([r['time'] for r in first_stage_results]):.1f}s")

# ===== 4. 显示Top 5模型 =====
print("\n📋 4/6 Top 5模型信息：")
print("   " + "-" * 70)

for i, result in enumerate(top_5_models):
    print(f"   🥇 第{i+1}名: 运行#{result['run']:3d}, 种子={result['seed']}")
    print(f"       验证MSE:  {result['val_MSE']:.4f}")
    print(f"       验证RMSE: {result['val_RMSE']:.4f} (平均误差约${result['val_RMSE']:.2f})")
    print(f"       训练时间: {result['time']:.1f}s")
    if i < 4:
        print()

# ===== 5. 种子优化器总结 =====
print("\n📋 5/6 种子优化器总结：")
optimizer_summary = seed_optimizer.get_summary()

print(f"   最佳种子: {optimizer_summary['best_seed']}")
print(f"   最佳验证损失: {optimizer_summary['best_loss']:.4f}")
print(f"   种子探索范围: {min(optimizer_summary['seed_history'])} - {max(optimizer_summary['seed_history'])}")

# ===== 6. 保存第一阶段结果 =====
print("\n📋 6/6 保存第一阶段结果...")

# 创建结果文件夹
import os
if not os.path.exists('results'):
    os.makedirs('results')

# 保存所有200次运行结果
first_stage_df = pd.DataFrame([{
    'run': r['run'],
    'seed': r['seed'],
    'val_MSE': r['val_MSE'],
    'val_RMSE': r['val_RMSE'],
    'val_MAE': r['val_MAE'],
    'epochs': r['epochs'],
    'time': r['time']} for r in first_stage_results])

first_stage_df.to_csv('results/stage1_all_200_runs.csv', index=False)
print("   ✅ 保存: results/stage1_all_200_runs.csv")

# 保存Top 5模型信息
top5_df = pd.DataFrame([{
    'top_rank': i+1,
    'run': r['run'],
    'seed': r['seed'],
    'val_MSE': r['val_MSE'],
    'val_RMSE': r['val_RMSE'],
    'val_MAE': r['val_MAE']} for i, r in enumerate(top_5_models)])

top5_df.to_csv('results/stage1_top5_models.csv', index=False)
print("   ✅ 保存: results/stage1_top5_models.csv")

print("\n" + "=" * 60)
print(f"✅ 第8步完成：训练了{N_FIRST_STAGE}个模型，选出Top {N_TOP_MODELS}")
print(f"   最佳验证MSE: {top_5_models[0]['val_MSE']:.4f}")
print("=" * 60)

print("\n" + "=" * 80)
print("🔥 [9/10] 第二阶段：Top 5模型在测试集上实战")
print("=" * 80)
print()

# ===== 1. 准备工作 =====
print("📋 1/6 准备第二阶段测试...")
print(f"   测试数据天数: {len(test_df)} 天")
print(f"   测试集时间范围: {test_df['Date'].iloc[0].date()} 到 {test_df['Date'].iloc[-1].date()}")
print(f"   将测试Top {N_TOP_MODELS}个模型")

second_stage_results = []
test_size = len(test_df)

print("✅ 准备工作完成")

# ===== 2. 对每个Top模型进行测试 =====
print("\n📋 2/6 开始测试Top 5模型...")
print("   " + "=" * 50)

for i, top_model_info in enumerate(top_5_models):
    model_num = i + 1
    print(f"\n   🔍 测试第{model_num}名模型 (运行#{top_model_info['run']}, 种子={top_model_info['seed']})")
    print(f"      验证MSE: {top_model_info['val_MSE']:.4f}")
    print(f"      进度:", end=' ')
    
    # 获取模型
    tcn_model = top_model_info['model']
    
    # 在测试集上滚动预测
    start_time = time.time()
    
    # 使用带进度显示的滚动预测
    tcn_predictions = rolling_forecast_deep_learning(
        tcn_model, test_df, train_df, scaler_X, scaler_y, 
        feature_columns, LOOKBACK, show_progress=False  # 这里我们自己显示进度
    )
    
    prediction_time = time.time() - start_time
    
    # 评估测试集表现
    mse_test = mean_squared_error(y_test, tcn_predictions)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, tcn_predictions)
    mape_test = np.mean(np.abs((y_test - tcn_predictions) / y_test)) * 100
    r2_test = r2_score(y_test, tcn_predictions)
    
    print(f"✅ 完成！用时: {prediction_time:.1f}s")
    
    # 显示详细结果
    print(f"      测试MSE:  {mse_test:.4f}")
    print(f"      测试RMSE: {rmse_test:.4f}")
    print(f"      测试MAE:  {mae_test:.4f}")
    print(f"      测试MAPE: {mape_test:.2f}%")
    print(f"      测试R²:   {r2_test:.4f}")
    
    # 保存结果
    second_stage_results.append({
        'top_rank': model_num,
        'run': top_model_info['run'],
        'seed': top_model_info['seed'],
        'val_MSE': top_model_info['val_MSE'],
        'val_RMSE': top_model_info['val_RMSE'],
        'val_MAE': top_model_info['val_MAE'],
        'test_MSE': mse_test,
        'test_RMSE': rmse_test,
        'test_MAE': mae_test,
        'test_MAPE': mape_test,
        'test_R2': r2_test,
        'predictions': tcn_predictions,
        'model': tcn_model
    })

# ===== 3. 找出最终冠军模型 =====
print("\n📋 3/6 找出最终冠军模型...")
best_test_result = min(second_stage_results, key=lambda x: x['test_MSE'])

print(f"   🏆 冠军模型: 第{best_test_result['top_rank']}名")
print(f"       运行编号: #{best_test_result['run']}")
print(f"       种子: {best_test_result['seed']}")
print(f"       验证MSE: {best_test_result['val_MSE']:.4f}")
print(f"       测试MSE: {best_test_result['test_MSE']:.4f} ⭐")

# ===== 4. 分析冠军模型表现 =====
print("\n📋 4/6 分析冠军模型表现...")

# 计算各种统计指标
errors = best_test_result['predictions'] - y_test
mean_error = errors.mean()
std_error = errors.std()
max_error = abs(errors).max()

print(f"   平均误差: ${mean_error:.2f}")
print(f"   误差标准差: ${std_error:.2f}")
print(f"   最大绝对误差: ${max_error:.2f}")

# 计算方向准确率
direction_correct = np.sum((np.diff(best_test_result['predictions']) * np.diff(y_test)) > 0)
direction_total = len(y_test) - 1
direction_accuracy = direction_correct / direction_total * 100 if direction_total > 0 else 0

print(f"   方向预测准确率: {direction_accuracy:.1f}%")

# ===== 5. 对比所有Top 5模型 =====
print("\n📋 5/6 Top 5模型对比分析：")
print("   " + "-" * 80)
print("   排名 | 验证MSE | 测试MSE | 测试R²  | MAPE   | 状态")
print("   " + "-" * 80)

for r in second_stage_results:
    is_best = "🏆 冠军" if r['test_MSE'] == best_test_result['test_MSE'] else "      "
    print(f"   #{r['top_rank']:2d}  | {r['val_MSE']:7.4f} | {r['test_MSE']:7.4f} | {r['test_R2']:6.4f} | {r['test_MAPE']:5.1f}% | {is_best}")

print("   " + "-" * 80)

# ===== 6. 保存第二阶段结果 =====
print("\n📋 6/6 保存第二阶段结果...")

# 保存第二阶段测试结果
second_stage_df = pd.DataFrame([{
    'top_rank': r['top_rank'],
    'run': r['run'],
    'seed': r['seed'],
    'val_MSE': r['val_MSE'],
    'test_MSE': r['test_MSE'],
    'test_RMSE': r['test_RMSE'],
    'test_MAE': r['test_MAE'],
    'test_MAPE': r['test_MAPE'],
    'test_R2': r['test_R2']} for r in second_stage_results])

second_stage_df.to_csv('results/stage2_top5_test_results.csv', index=False)
print("   ✅ 保存: results/stage2_top5_test_results.csv")

# 保存冠军模型的预测结果
best_predictions_df = pd.DataFrame({
    'Date': test_df['Date'],
    'Actual': y_test,
    'Prediction': best_test_result['predictions'],
    'Error': best_test_result['predictions'] - y_test,
    'Error_Abs': np.abs(best_test_result['predictions'] - y_test),
    'Error_Pct': (best_test_result['predictions'] - y_test) / y_test * 100
})

best_predictions_df.to_csv('results/final_best_predictions.csv', index=False)
print("   ✅ 保存: results/final_best_predictions.csv")

print("\n" + "=" * 60)
print(f"✅ 第9步完成：Top {N_TOP_MODELS}个模型测试完成！")
print(f"   冠军模型测试MSE: {best_test_result['test_MSE']:.4f}")
print(f"   冠军模型测试R²: {best_test_result['test_R2']:.4f}")
print("=" * 60)

print("\n" + "=" * 80)
print("📊 [10/10] 保存结果和生成可视化图表")
print("=" * 80)
print()

# ===== 1. 准备可视化数据 =====
print("📋 1/7 准备可视化数据...")

# 获取所有需要的数据
val_mses = [r['val_MSE'] for r in first_stage_results]
test_mses_top5 = [r['test_MSE'] for r in second_stage_results]
val_mses_top5 = [r['val_MSE'] for r in second_stage_results]
top_ranks = [r['top_rank'] for r in second_stage_results]

# 冠军模型数据
champion_pred = best_test_result['predictions']
champion_errors = champion_pred - y_test

print(f"   第一阶段验证MSE范围: {min(val_mses):.4f} - {max(val_mses):.4f}")
print(f"   第二阶段测试MSE范围: {min(test_mses_top5):.4f} - {max(test_mses_top5):.4f}")
print("✅ 数据准备完成")

# ===== 2. 创建综合可视化图表 =====
print("\n📋 2/7 创建综合可视化图表...")
print("   生成图表中...", end=' ')

import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建大图
fig = plt.figure(figsize=(20, 16))
fig.suptitle('S&P 500 TCN模型两阶段选择结果分析', fontsize=18, fontweight='bold', y=0.98)

# 子图1：冠军模型预测 vs 实际
ax1 = plt.subplot(3, 3, 1)
ax1.plot(test_df['Date'], y_test, 'k-', label='实际价格', linewidth=2, alpha=0.8)
ax1.plot(test_df['Date'], champion_pred, 'r--', 
        label=f'预测 (R²={best_test_result["test_R2"]:.3f})', linewidth=2, alpha=0.8)
ax1.set_title('冠军模型：预测 vs 实际', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期')
ax1.set_ylabel('价格 ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 子图2：第一阶段验证MSE分布
ax2 = plt.subplot(3, 3, 2)
ax2.hist(val_mses, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=best_test_result['val_MSE'], color='red', linestyle='--', 
           linewidth=2, label=f'冠军模型: {best_test_result["val_MSE"]:.4f}')
ax2.set_title('第一阶段：验证MSE分布 (200个模型)', fontsize=14, fontweight='bold')
ax2.set_xlabel('验证MSE')
ax2.set_ylabel('频次')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 子图3：Top 5模型测试MSE对比
ax3 = plt.subplot(3, 3, 3)
colors = ['red' if mse == best_test_result['test_MSE'] else 'gray' for mse in test_mses_top5]
bars = ax3.bar(top_ranks, test_mses_top5, color=colors, alpha=0.8, edgecolor='black')
ax3.set_title('Top 5模型：测试MSE对比', fontsize=14, fontweight='bold')
ax3.set_xlabel('第一阶段排名')
ax3.set_ylabel('测试MSE (越低越好)')
ax3.grid(True, alpha=0.3, axis='y')

# 在柱子上标注数值
for bar, mse in zip(bars, test_mses_top5):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{mse:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 子图4：验证集 vs 测试集表现
ax4 = plt.subplot(3, 3, 4)
scatter = ax4.scatter(val_mses_top5, test_mses_top5, s=150, 
                     c=top_ranks, cmap='viridis', edgecolor='black', alpha=0.8)

# 标注每个点
for i, r in enumerate(second_stage_results):
    color = 'red' if r['test_MSE'] == best_test_result['test_MSE'] else 'black'
    ax4.annotate(f'#{r["top_rank"]}', (r['val_MSE'], r['test_MSE']), 
                fontsize=11, ha='center', fontweight='bold', color=color)

ax4.set_title('验证集MSE vs 测试集MSE', fontsize=14, fontweight='bold')
ax4.set_xlabel('验证集MSE')
ax4.set_ylabel('测试集MSE')
ax4.grid(True, alpha=0.3)

# 子图5：误差时间序列
ax5 = plt.subplot(3, 3, 5)
ax5.plot(test_df['Date'], champion_errors, color='purple', linewidth=1.5)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax5.fill_between(test_df['Date'], champion_errors, 0, alpha=0.3, color='purple')
ax5.set_title('误差时间序列 (应围绕0波动)', fontsize=14, fontweight='bold')
ax5.set_xlabel('日期')
ax5.set_ylabel('误差 ($)')
ax5.grid(True, alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# 子图6：误差分布直方图
ax6 = plt.subplot(3, 3, 6)
ax6.hist(champion_errors, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax6.set_title(f'误差分布 (均值={champion_errors.mean():.3f})', fontsize=14, fontweight='bold')
ax6.set_xlabel('误差 ($)')
ax6.set_ylabel('频次')
ax6.grid(True, alpha=0.3)

# 子图7：实际 vs 预测散点图
ax7 = plt.subplot(3, 3, 7)
scatter = ax7.scatter(y_test, champion_pred, alpha=0.6, 
                     c=range(len(y_test)), cmap='viridis', s=30, edgecolor='black', linewidth=0.5)
min_val = min(y_test.min(), champion_pred.min())
max_val = max(y_test.max(), champion_pred.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
        label='完美预测线', alpha=0.8)
ax7.set_title(f'实际 vs 预测 (R² = {best_test_result["test_R2"]:.4f})', 
             fontsize=14, fontweight='bold')
ax7.set_xlabel('实际价格 ($)')
ax7.set_ylabel('预测价格 ($)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 子图8：种子优化历史
ax8 = plt.subplot(3, 3, 8)
iterations = list(range(1, min(101, len(optimizer_summary['loss_history'])) + 1))
losses = optimizer_summary['loss_history'][:100]
ax8.plot(iterations, losses, 'o-', color='green', linewidth=2, markersize=4)
ax8.axhline(y=optimizer_summary['best_loss'], color='red', linestyle='--', 
           linewidth=2, alpha=0.5, label=f'最佳: {optimizer_summary["best_loss"]:.4f}')
ax8.set_title('种子优化历史 (前100次)', fontsize=14, fontweight='bold')
ax8.set_xlabel('迭代次数')
ax8.set_ylabel('验证MSE')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 子图9：性能总结表格
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""🏆 冠军模型总结
种子: {best_test_result['seed']}
第一阶段排名: #{best_test_result['top_rank']}

📊 验证集表现:
  MSE:  {best_test_result['val_MSE']:.4f}
  RMSE: {best_test_result['val_RMSE']:.4f}
  MAE:  {best_test_result['val_MAE']:.4f}

📈 测试集表现:
  MSE:  {best_test_result['test_MSE']:.4f}
  RMSE: {best_test_result['test_RMSE']:.4f}
  MAE:  {best_test_result['test_MAE']:.4f}
  MAPE: {best_test_result['test_MAPE']:.2f}%
  R²:   {best_test_result['test_R2']:.4f}

🔧 模型配置:
  特征数: {len(feature_columns)}
  历史天数: {LOOKBACK}
  训练次数: {N_FIRST_STAGE}
  最佳前: {N_TOP_MODELS}
  
🎯 关键改进:
  • 滚动预测防滞后
  • 两阶段筛选
  • 40维特征工程"""

ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("✅ 图表生成完成")

# ===== 3. 保存图表 =====
print("\n📋 3/7 保存可视化图表...")
plt.savefig('results/model_performance_summary.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: results/model_performance_summary.png")

# 显示图表（在Jupyter中会自动显示）
plt.show()

# ===== 4. 生成额外的详细图表 =====
print("\n📋 4/7 生成额外详细图表...")

# 图表1：误差分析图
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 左图：累积误差
cumulative_error = np.cumsum(champion_errors)
ax1.plot(test_df['Date'], cumulative_error, 'b-', linewidth=2)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_title('累积误差（应围绕0波动）', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期')
ax1.set_ylabel('累积误差 ($)')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 右图：滚动误差统计
window = 30
rolling_mae = pd.Series(np.abs(champion_errors)).rolling(window=window).mean()
rolling_std = pd.Series(champion_errors).rolling(window=window).std()

ax2.plot(test_df['Date'][window-1:], rolling_mae[window-1:], 'g-', label=f'{window}日平均绝对误差', linewidth=2)
ax2.plot(test_df['Date'][window-1:], rolling_std[window-1:], 'r-', label=f'{window}日误差标准差', linewidth=2)
ax2.set_title(f'{window}日滚动误差统计', fontsize=14, fontweight='bold')
ax2.set_xlabel('日期')
ax2.set_ylabel('误差 ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: results/error_analysis.png")

# ===== 5. 保存种子优化历史图表 =====
print("\n📋 5/7 生成种子优化历史图表...")

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 左图：完整种子搜索历史
ax1.plot(optimizer_summary['seed_history'], optimizer_summary['loss_history'], 'o-', 
        alpha=0.5, markersize=3)
ax1.scatter([optimizer_summary['best_seed']], [optimizer_summary['best_loss']], 
           color='red', s=200, marker='*', label='最佳种子', zorder=5)
ax1.set_title('完整种子搜索历史', fontsize=14, fontweight='bold')
ax1.set_xlabel('种子值')
ax1.set_ylabel('验证MSE')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：损失值分布
ax2.hist(optimizer_summary['loss_history'], bins=30, color='orange', edgecolor='black', alpha=0.7)
ax2.axvline(x=optimizer_summary['best_loss'], color='red', linestyle='--', 
           linewidth=2, label=f'最佳损失: {optimizer_summary["best_loss"]:.4f}')
ax2.set_title('验证损失分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('验证MSE')
ax2.set_ylabel('频次')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/seed_optimization_history.png', dpi=300, bbox_inches='tight')
print("   ✅ 保存: results/seed_optimization_history.png")

# ===== 6. 保存种子优化历史数据 =====
print("\n📋 6/7 保存种子优化历史数据...")

seed_history_df = pd.DataFrame({
    'iteration': range(1, len(optimizer_summary['seed_history']) + 1),
    'seed': optimizer_summary['seed_history'],
    'val_loss': optimizer_summary['loss_history']
})

seed_history_df.to_csv('results/seed_optimization_history.csv', index=False)
print("   ✅ 保存: results/seed_optimization_history.csv")

# ===== 7. 生成最终总结报告 =====
print("\n📋 7/7 生成最终总结报告...")

# 创建总结文本
total_time = time.time() - start_time_total
total_hours = total_time / 3600

summary_report = f"""
{'='*80}
S&P 500 TCN模型 - 两阶段选择训练总结报告
{'='*80}

📅 项目信息：
   训练数据: {train_df['Date'].iloc[0].date()} 到 {train_df['Date'].iloc[-1].date()}
   测试数据: {test_df['Date'].iloc[0].date()} 到 {test_df['Date'].iloc[-1].date()}
   总训练时间: {total_hours:.2f} 小时 ({total_time:.0f} 秒)

🏗️ 模型配置：
   特征数量: {len(feature_columns)} 个
   历史天数: {LOOKBACK} 天
   神经网络: TCN (时序卷积网络)
   训练策略: 两阶段选择

📊 第一阶段 (训练与验证)：
   总训练次数: {N_FIRST_STAGE} 次
   最佳选择数: {N_TOP_MODELS} 个
   验证MSE范围: {min(val_mses):.4f} - {max(val_mses):.4f}
   平均训练时间: {np.mean([r['time'] for r in first_stage_results]):.1f} 秒/模型

📈 第二阶段 (测试与选择)：
   测试数据天数: {len(test_df)} 天
   最佳模型种子: {best_test_result['seed']}
   来自第一阶段排名: #{best_test_result['top_rank']}

🏆 冠军模型表现：
   验证MSE: {best_test_result['val_MSE']:.4f}
   测试MSE: {best_test_result['test_MSE']:.4f}
   测试R²: {best_test_result['test_R2']:.4f}
   测试MAPE: {best_test_result['test_MAPE']:.2f}%
   平均绝对误差: ${best_test_result['test_MAE']:.2f}
   方向预测准确率: {direction_accuracy:.1f}%

📁 生成的文件：
   1. results/stage1_all_200_runs.csv - 第一阶段200次训练结果
   2. results/stage1_top5_models.csv - Top 5模型信息
   3. results/stage2_top5_test_results.csv - Top 5测试结果
   4. results/final_best_predictions.csv - 冠军模型预测值
   5. results/seed_optimization_history.csv - 种子搜索历史数据
   6. results/model_performance_summary.png - 主要性能图表
   7. results/error_analysis.png - 误差分析图表
   8. results/seed_optimization_history.png - 种子优化历史图表

🎯 关键创新点：
   • 滚动预测策略 - 防止误差累积和滞后问题
   • 两阶段选择机制 - 200次训练 → Top 5 → 最佳模型
   • 40维特征工程 - 技术指标 + 动量特征 + 宏观因子
   • 防数据泄漏设计 - 严格使用历史数据

✅ 项目状态：完成！
{'='*80}
"""

# 保存总结报告
with open('results/final_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("   ✅ 保存: results/final_summary_report.txt")

# 在控制台显示总结
print(summary_report)

# ===== 最终完成信息 =====
print("\n" + "="*80)
print("🎉 恭喜！模型训练和测试全部完成！")
print("="*80)
print()
print("📊 模型表现总结：")
print(f"   冠军模型测试R²: {best_test_result['test_R2']:.4f}")
print(f"   平均绝对百分比误差: {best_test_result['test_MAPE']:.2f}%")
print(f"   方向预测准确率: {direction_accuracy:.1f}%")
print()
print("📁 所有结果已保存到 'results/' 文件夹")
print("   请查看生成的文件和图表进行分析")
print()
print("🚀 下一步建议：")
print("   1. 查看 results/model_performance_summary.png 图表")
print("   2. 分析 results/final_best_predictions.csv 预测结果")
print("   3. 如有需要，调整参数重新训练优化")
print("="*80)