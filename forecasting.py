import numpy as np
import pandas as pd
from feature_engineering import create_features_no_leakage


def rolling_forecast_deep_learning(model, test_df, train_df,
                                    scaler_X, scaler_y, feature_columns,
                                    lookback, show_progress=True):
    """
    TCN 模型滚动预测：每步用真实值更新历史，防止误差累积。
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
            filled  = int(40 * i // test_size)
            bar     = '█' * filled + '░' * (40 - filled)
            print(f"   [{bar}] {i:4d}/{test_size} ({percent:5.1f}%)", end='\r')

        try:
            current_history = create_features_no_leakage(history_df)
            current_history = current_history.bfill().ffill()
            recent_features = current_history[feature_columns].values[-lookback:]
            recent_scaled   = scaler_X.transform(recent_features)
            X_input         = recent_scaled.reshape(1, lookback, len(feature_columns))
            pred_scaled     = model.predict(X_input, verbose=0)[0, 0]
            pred            = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
        except Exception:
            predictions.append(predictions[-1] if predictions else history_df['Close/Last'].iloc[-1])

        next_row   = test_df.iloc[i:i + 1].copy()
        history_df = pd.concat([history_df, next_row], ignore_index=True)

    if show_progress:
        print(f"   [{'█' * 40}] {test_size:4d}/{test_size} (100.0%) ✅\n")

    return np.array(predictions)
