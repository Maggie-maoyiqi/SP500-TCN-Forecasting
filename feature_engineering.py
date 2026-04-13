import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def create_features_no_leakage(df):
    """创建特征，严格避免数据泄露（全部使用前一天数据）"""
    df = df.copy()

    # ===== 基础特征 =====
    df['prev_close']  = df['Close/Last'].shift(1)
    df['prev_open']   = df['Open'].shift(1)
    df['prev_high']   = df['High'].shift(1)
    df['prev_low']    = df['Low'].shift(1)
    df['prev_volume'] = df['Volume'].shift(1)

    # ===== 技术指标 =====
    df['prev_hl_range'] = df['prev_high'] - df['prev_low']
    df['prev_oc_diff']  = df['prev_open'] - df['prev_close']
    df['returns_1d']    = df['Close/Last'].pct_change(1).shift(1)
    df['returns_5d']    = df['Close/Last'].pct_change(5).shift(1)

    for window in [5, 10, 20]:
        ma = df['Close/Last'].shift(1).rolling(window).mean()
        df[f'MA_{window}']       = ma
        df[f'MA_{window}_ratio'] = df['prev_close'] / ma

    delta = df['Close/Last'].diff().shift(1)
    gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    exp1 = df['Close/Last'].shift(1).ewm(span=12, adjust=False).mean()
    exp2 = df['Close/Last'].shift(1).ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    df['volatility_10'] = df['returns_1d'].rolling(10).std()
    volume_ma = df['Volume'].shift(1).rolling(10).mean()
    df['volume_ratio'] = df['prev_volume'] / volume_ma

    # ===== 动量特征 =====
    df['momentum_2d'] = df['Close/Last'].shift(1) - df['Close/Last'].shift(3)
    df['momentum_3d'] = df['Close/Last'].shift(1) - df['Close/Last'].shift(4)
    df['momentum_5d'] = df['Close/Last'].shift(1) - df['Close/Last'].shift(6)
    df['acceleration'] = df['returns_1d'] - df['returns_1d'].shift(1)

    ema_fast = df['Close/Last'].shift(1).ewm(span=5,  adjust=False).mean()
    ema_slow = df['Close/Last'].shift(1).ewm(span=20, adjust=False).mean()
    df['trend_strength'] = (ema_fast - ema_slow) / ema_slow

    short_momentum = df['Close/Last'].shift(1) - df['Close/Last'].shift(3)
    long_momentum  = df['Close/Last'].shift(1) - df['Close/Last'].shift(11)
    df['momentum_ratio']     = short_momentum / (long_momentum.abs() + 1e-8)
    df['volatility_momentum'] = df['volatility_10'] - df['volatility_10'].shift(5)
    df['volume_momentum']    = (df['Volume'].shift(1) - df['Volume'].shift(6)) / df['Volume'].shift(6)
    df['ROC_5']  = (df['Close/Last'].shift(1) - df['Close/Last'].shift(6))  / df['Close/Last'].shift(6)  * 100
    df['ROC_10'] = (df['Close/Last'].shift(1) - df['Close/Last'].shift(11)) / df['Close/Last'].shift(11) * 100

    rolling_mean = df['Close/Last'].shift(1).rolling(20).mean()
    rolling_std  = df['Close/Last'].shift(1).rolling(20).std()
    df['bb_position'] = (df['prev_close'] - rolling_mean) / (2 * rolling_std)

    # ===== 价量因子 =====
    df['intraday_momentum'] = (df['prev_close'] - df['prev_open']) / (df['prev_high'] - df['prev_low'] + 1e-8)
    df['candle_body_ratio'] = np.abs(df['prev_close'] - df['prev_open']) / (df['prev_high'] - df['prev_low'] + 1e-8)
    df['normalized_range']  = (df['prev_high'] - df['prev_low']) / (df['prev_open'] + 1e-8)
    df['price_volume_fit']  = (df['prev_close'] - df['prev_open']) * df['prev_volume']
    df['money_flow'] = (((df['prev_close'] - df['prev_low']) - (df['prev_high'] - df['prev_close'])) /
                        (df['prev_high'] - df['prev_low'] + 1e-8)) * df['prev_volume']

    # ===== 宏观经济因子 =====
    df['prev_dgs10']   = df['DGS10'].shift(1)
    df['prev_gdpc1']   = df['GDPC1'].shift(1)
    df['dgs10_change'] = df['DGS10'].diff().shift(1)
    df['dgs10_ma_10']  = df['DGS10'].shift(1).rolling(10).mean()
    df['gdpc1_growth'] = df['GDPC1'].pct_change(1).shift(1)

    return df


def get_feature_columns(df):
    """返回特征列名列表（排除原始列）"""
    exclude = ['Date', 'Close/Last', 'Open', 'High', 'Low', 'Volume', 'DGS10', 'GDPC1']
    return [col for col in df.columns if col not in exclude]


def prepare_sequences(train_df, lookback):
    """
    对训练数据做特征工程、归一化、滑动窗口切分。
    返回: X_train, y_train, X_val, y_val, scaler_X, scaler_y,
          feature_columns, train_with_features
    """
    train_with_features = create_features_no_leakage(train_df)
    train_with_features = train_with_features.bfill().ffill()

    feature_columns = get_feature_columns(train_with_features)

    X_data = train_with_features[feature_columns].values
    y_data = train_with_features['Close/Last'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))

    Xs, ys = [], []
    for i in tqdm(range(lookback, len(X_scaled)), desc="创建序列", leave=False):
        Xs.append(X_scaled[i - lookback:i])
        ys.append(y_scaled[i])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    split_idx = int(len(X_seq) * 0.85)
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_val,   y_val   = X_seq[split_idx:], y_seq[split_idx:]

    return X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_columns, train_with_features
