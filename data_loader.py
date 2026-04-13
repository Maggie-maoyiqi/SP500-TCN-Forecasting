import pandas as pd


def load_data(train_path, test_path, dgs10_path, gdpc1_path):
    """加载原始 CSV 数据，返回 train_df, test_df, dgs10_df, gdpc1_df"""
    train_df  = pd.read_csv(train_path)
    test_df   = pd.read_csv(test_path)
    dgs10_df  = pd.read_csv(dgs10_path)
    gdpc1_df  = pd.read_csv(gdpc1_path)

    # 日期格式转换
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date']  = pd.to_datetime(test_df['Date'])
    dgs10_df['observation_date'] = pd.to_datetime(dgs10_df['observation_date'])
    gdpc1_df['observation_date'] = pd.to_datetime(gdpc1_df['observation_date'])

    # 按日期排序
    train_df = train_df.sort_values('Date').reset_index(drop=True)
    test_df  = test_df.sort_values('Date').reset_index(drop=True)

    # 清理价格数据（去掉美元符号）
    for col in ['Close/Last', 'Open', 'High', 'Low']:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].str.replace('$', '', regex=False).astype(float)
            test_df[col]  = test_df[col].str.replace('$', '', regex=False).astype(float)

    return train_df, test_df, dgs10_df, gdpc1_df


def merge_macro_data(train_df, test_df, dgs10_df, gdpc1_df):
    """合并宏观经济数据，返回合并后的 train_df, test_df"""
    dgs10_df = dgs10_df.copy()
    gdpc1_df = gdpc1_df.copy()

    dgs10_df.columns = ['Date', 'DGS10']
    gdpc1_df.columns = ['Date', 'GDPC1']

    dgs10_df['DGS10'] = pd.to_numeric(dgs10_df['DGS10'], errors='coerce')
    gdpc1_df['GDPC1'] = pd.to_numeric(gdpc1_df['GDPC1'], errors='coerce')

    train_df = train_df.merge(dgs10_df, on='Date', how='left')
    test_df  = test_df.merge(dgs10_df, on='Date', how='left')
    train_df = train_df.merge(gdpc1_df, on='Date', how='left')
    test_df  = test_df.merge(gdpc1_df, on='Date', how='left')

    # 向前/向后填充缺失的宏观数据
    for df in [train_df, test_df]:
        df['DGS10'] = df['DGS10'].ffill().bfill()
        df['GDPC1'] = df['GDPC1'].ffill().bfill()

    return train_df, test_df
