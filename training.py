import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import build_tcn_model
from forecasting import rolling_forecast_deep_learning


def run_stage1(X_train, y_train, X_val, y_val, scaler_y,
               input_shape, seed_optimizer, n_first_stage, n_top_models):
    """
    第一阶段：训练 n_first_stage 个模型，按验证 MSE 排序，返回 Top n_top_models。
    返回: first_stage_results, top_models, seed_optimizer
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)

    first_stage_results = []
    start_time_total    = time.time()

    print(f"📋 开始第一阶段：训练 {n_first_stage} 个模型...")
    print("   " + "=" * 55)

    for run in range(n_first_stage):
        # 进度条
        if run % max(1, n_first_stage // 50) == 0 or run == n_first_stage - 1:
            percent = (run / n_first_stage) * 100
            filled  = int(50 * run // n_first_stage)
            bar     = '█' * filled + '░' * (50 - filled)
            elapsed = time.time() - start_time_total
            if run > 0:
                remaining = (elapsed / run) * (n_first_stage - run)
                time_str  = f"剩余: {remaining:.0f}s"
            else:
                time_str = "计算中..."
            print(f"   [{bar}] {run:3d}/{n_first_stage} ({percent:5.1f}%) {time_str}", end='\r')

        # 设置随机种子
        current_seed = seed_optimizer.get_next_seed()
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)

        # 构建并训练模型
        t0 = time.time()
        tcn_model = build_tcn_model(input_shape, verbose=False)
        history   = tcn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # 在验证集评估
        val_pred          = tcn_model.predict(X_val, verbose=0)
        val_pred_original = scaler_y.inverse_transform(val_pred)
        val_true_original = scaler_y.inverse_transform(y_val.reshape(-1, 1))

        val_mse  = mean_squared_error(val_true_original, val_pred_original)
        val_rmse = np.sqrt(val_mse)
        val_mae  = mean_absolute_error(val_true_original, val_pred_original)

        seed_optimizer.update(current_seed, val_mse)

        first_stage_results.append({
            'run':      run + 1,
            'seed':     current_seed,
            'val_MSE':  val_mse,
            'val_RMSE': val_rmse,
            'val_MAE':  val_mae,
            'epochs':   len(history.history['loss']),
            'time':     time.time() - t0,
            'model':    tcn_model,
        })

    elapsed_total = time.time() - start_time_total
    print(f"   [{'█' * 50}] {n_first_stage}/{n_first_stage} (100.0%) "
          f"完成！总用时: {elapsed_total:.0f}s")

    first_stage_results_sorted = sorted(first_stage_results, key=lambda x: x['val_MSE'])
    top_models = first_stage_results_sorted[:n_top_models]

    print(f"\n✅ 第一阶段完成：训练了 {n_first_stage} 个模型，选出 Top {n_top_models}")
    print(f"   验证 MSE 范围: {first_stage_results_sorted[0]['val_MSE']:.4f} "
          f"- {first_stage_results_sorted[-1]['val_MSE']:.4f}")
    print(f"   平均训练时间: {np.mean([r['time'] for r in first_stage_results]):.1f}s/模型\n")

    return first_stage_results, top_models, seed_optimizer


def run_stage2(top_models, test_df, train_df, scaler_X, scaler_y,
               feature_columns, lookback, y_test):
    """
    第二阶段：对 Top 模型做测试集滚动预测，找出冠军。
    返回: second_stage_results, best_test_result
    """
    second_stage_results = []

    print(f"📋 开始第二阶段：测试 Top {len(top_models)} 个模型...")
    print("   " + "=" * 55)

    for i, top_model_info in enumerate(top_models):
        model_num = i + 1
        print(f"\n   🔍 测试第 {model_num} 名模型 "
              f"(运行 #{top_model_info['run']}, 种子={top_model_info['seed']})")
        print(f"      验证 MSE: {top_model_info['val_MSE']:.4f}")

        t0 = time.time()
        tcn_predictions = rolling_forecast_deep_learning(
            top_model_info['model'], test_df, train_df,
            scaler_X, scaler_y, feature_columns, lookback,
            show_progress=False
        )
        prediction_time = time.time() - t0

        mse_test  = mean_squared_error(y_test, tcn_predictions)
        rmse_test = np.sqrt(mse_test)
        mae_test  = mean_absolute_error(y_test, tcn_predictions)
        mape_test = np.mean(np.abs((y_test - tcn_predictions) / y_test)) * 100
        r2_test   = r2_score(y_test, tcn_predictions)

        print(f"      ✅ 用时: {prediction_time:.1f}s  |  "
              f"MSE: {mse_test:.4f}  RMSE: {rmse_test:.4f}  "
              f"MAE: {mae_test:.4f}  MAPE: {mape_test:.2f}%  R²: {r2_test:.4f}")

        second_stage_results.append({
            'top_rank':  model_num,
            'run':       top_model_info['run'],
            'seed':      top_model_info['seed'],
            'val_MSE':   top_model_info['val_MSE'],
            'val_RMSE':  top_model_info['val_RMSE'],
            'val_MAE':   top_model_info['val_MAE'],
            'test_MSE':  mse_test,
            'test_RMSE': rmse_test,
            'test_MAE':  mae_test,
            'test_MAPE': mape_test,
            'test_R2':   r2_test,
            'predictions': tcn_predictions,
            'model':       top_model_info['model'],
        })

    best_test_result = min(second_stage_results, key=lambda x: x['test_MSE'])

    print(f"\n✅ 第二阶段完成！")
    print(f"   🏆 冠军：第 {best_test_result['top_rank']} 名  "
          f"种子={best_test_result['seed']}  "
          f"测试 MSE={best_test_result['test_MSE']:.4f}  "
          f"R²={best_test_result['test_R2']:.4f}\n")

    return second_stage_results, best_test_result
