"""
S&P 500 TCN 预测 - 主入口
运行方式: python main.py
所有图表和 CSV 结果保存至 results/，训练日志同步打印到终端并保存为 txt。
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ===== Logger：同时输出到终端和文件 =====
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ===== 导入各模块 =====
from config import (TRAIN_PATH, TEST_PATH, DGS10_PATH, GDPC1_PATH,
                    SEED, LOOKBACK, N_FIRST_STAGE, N_TOP_MODELS, RESULTS_DIR)
from data_loader import load_data, merge_macro_data
from feature_engineering import prepare_sequences
from model import build_tcn_model
from seed_optimizer import GradientBasedSeedOptimizer
from forecasting import rolling_forecast_deep_learning
from training import run_stage1, run_stage2
from visualization import (create_performance_chart,
                            create_error_analysis_chart,
                            create_seed_history_chart)


# ───────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 启动日志（所有 print 同时写入 txt）
    logger = Logger(os.path.join(RESULTS_DIR, 'training_log.txt'))
    sys.stdout = logger

    total_start = time.time()

    # ──────────────────────────────────────────────────────
    # 步骤 1：加载数据
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📂 [1/10] 加载数据")
    print("=" * 80)

    train_df, test_df, dgs10_df, gdpc1_df = load_data(
        TRAIN_PATH, TEST_PATH, DGS10_PATH, GDPC1_PATH)

    print(f"   ✓ 训练数据：{len(train_df)} 行")
    print(f"   ✓ 测试数据：{len(test_df)} 行")
    print(f"   ✓ 国债数据：{len(dgs10_df)} 行")
    print(f"   ✓ GDP数据： {len(gdpc1_df)} 行")

    # ──────────────────────────────────────────────────────
    # 步骤 2：合并宏观经济数据
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🔗 [2/10] 合并宏观经济数据")
    print("=" * 80)

    train_df, test_df = merge_macro_data(train_df, test_df, dgs10_df, gdpc1_df)
    print("   ✓ 合并了 DGS10（10年期国债收益率）")
    print("   ✓ 合并了 GDPC1（实际GDP）")
    print("   ✓ 使用前向填充处理了缺失值")

    # ──────────────────────────────────────────────────────
    # 步骤 3 & 4：特征工程 + 数据预处理
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("⚙️  [3/10] 特征工程 + [4/10] 数据预处理")
    print("=" * 80)

    (X_train, y_train, X_val, y_val,
     scaler_X, scaler_y, feature_columns,
     train_with_features) = prepare_sequences(train_df, LOOKBACK)

    print(f"   ✓ 创建了 {len(feature_columns)} 个特征")
    print(f"   ✓ 训练集：{len(X_train)} 个样本")
    print(f"   ✓ 验证集：{len(X_val)} 个样本")
    print(f"   ✓ 输入形状：{X_train.shape}  (样本数, 时间步长, 特征数)")

    # ──────────────────────────────────────────────────────
    # 步骤 5：构建 TCN 模型（测试用）
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🏗️  [5/10] 构建 TCN 模型")
    print("=" * 80)

    input_shape = (LOOKBACK, len(feature_columns))
    test_model  = build_tcn_model(input_shape, verbose=True)
    print(f"   ✓ 模型构建成功，总参数: {test_model.count_params():,}")

    # ──────────────────────────────────────────────────────
    # 步骤 6：初始化种子优化器
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🎲 [6/10] 初始化梯度种子优化器")
    print("=" * 80)

    seed_optimizer = GradientBasedSeedOptimizer(base_seed=SEED, learning_rate=15.0)
    print(f"   ✓ 基础种子: {SEED}  学习率: 15.0")

    # ──────────────────────────────────────────────────────
    # 步骤 7：测试滚动预测函数（小样本）
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🔄 [7/10] 测试滚动预测函数（前5天）")
    print("=" * 80)

    try:
        test_preds = rolling_forecast_deep_learning(
            test_model, test_df.head(5), train_df,
            scaler_X, scaler_y, feature_columns, LOOKBACK, show_progress=True)
        print(f"   ✓ 滚动预测函数正常，预测范围: "
              f"${test_preds.min():.2f} - ${test_preds.max():.2f}")
    except Exception as e:
        print(f"   ⚠ 测试跳过: {e}")

    # ──────────────────────────────────────────────────────
    # 步骤 8：第一阶段训练（200个模型）
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"🔥 [8/10] 第一阶段：训练 {N_FIRST_STAGE} 个模型，选 Top {N_TOP_MODELS}")
    print("=" * 80)

    first_stage_results, top_models, seed_optimizer = run_stage1(
        X_train, y_train, X_val, y_val, scaler_y,
        input_shape, seed_optimizer, N_FIRST_STAGE, N_TOP_MODELS)

    # 保存第一阶段 CSV
    pd.DataFrame([{k: v for k, v in r.items() if k != 'model'}
                  for r in first_stage_results]).to_csv(
        os.path.join(RESULTS_DIR, 'stage1_all_runs.csv'), index=False)
    pd.DataFrame([{'top_rank': i + 1, 'run': r['run'], 'seed': r['seed'],
                   'val_MSE': r['val_MSE'], 'val_RMSE': r['val_RMSE'],
                   'val_MAE': r['val_MAE']}
                  for i, r in enumerate(top_models)]).to_csv(
        os.path.join(RESULTS_DIR, f'stage1_top{N_TOP_MODELS}_models.csv'), index=False)
    print(f"   ✅ 保存: {RESULTS_DIR}/stage1_all_runs.csv")
    print(f"   ✅ 保存: {RESULTS_DIR}/stage1_top{N_TOP_MODELS}_models.csv")

    # ──────────────────────────────────────────────────────
    # 步骤 9：第二阶段测试（Top 模型实战）
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"🔥 [9/10] 第二阶段：Top {N_TOP_MODELS} 模型在测试集上实战")
    print("=" * 80)

    y_test = test_df['Close/Last'].values
    second_stage_results, best_test_result = run_stage2(
        top_models, test_df, train_df, scaler_X, scaler_y,
        feature_columns, LOOKBACK, y_test)

    # 保存第二阶段 CSV
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('predictions', 'model')}
                  for r in second_stage_results]).to_csv(
        os.path.join(RESULTS_DIR, 'stage2_top_test_results.csv'), index=False)
    pd.DataFrame({
        'Date':      test_df['Date'].values,
        'Actual':    y_test,
        'Predicted': best_test_result['predictions'],
        'Error':     best_test_result['predictions'] - y_test,
        'Error_Abs': np.abs(best_test_result['predictions'] - y_test),
        'Error_Pct': (best_test_result['predictions'] - y_test) / y_test * 100,
    }).to_csv(os.path.join(RESULTS_DIR, 'final_best_predictions.csv'), index=False)

    optimizer_summary = seed_optimizer.get_summary()
    pd.DataFrame({
        'iteration': range(1, len(optimizer_summary['seed_history']) + 1),
        'seed':      optimizer_summary['seed_history'],
        'val_loss':  optimizer_summary['loss_history'],
    }).to_csv(os.path.join(RESULTS_DIR, 'seed_optimization_history.csv'), index=False)

    print(f"   ✅ 保存: {RESULTS_DIR}/stage2_top_test_results.csv")
    print(f"   ✅ 保存: {RESULTS_DIR}/final_best_predictions.csv")
    print(f"   ✅ 保存: {RESULTS_DIR}/seed_optimization_history.csv")

    # ──────────────────────────────────────────────────────
    # 步骤 10：可视化 & 文字总结
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 [10/10] 生成图表 & 汇总报告")
    print("=" * 80)

    # 计算冠军模型统计
    errors           = best_test_result['predictions'] - y_test
    direction_ok     = np.sum((np.diff(best_test_result['predictions']) * np.diff(y_test)) > 0)
    direction_acc    = direction_ok / (len(y_test) - 1) * 100

    # 三张图表
    champion_errors = create_performance_chart(
        test_df, y_test, best_test_result,
        first_stage_results, second_stage_results,
        optimizer_summary, feature_columns,
        LOOKBACK, N_FIRST_STAGE, N_TOP_MODELS, RESULTS_DIR)
    create_error_analysis_chart(test_df, champion_errors, RESULTS_DIR)
    create_seed_history_chart(optimizer_summary, RESULTS_DIR)

    # 汇总报告（同时经 Logger 写入 training_log.txt）
    total_time = time.time() - total_start
    report = f"""
{'=' * 80}
S&P 500 TCN 模型 - 训练总结报告
{'=' * 80}

📅 数据信息:
   训练: {train_df['Date'].iloc[0].date()} → {train_df['Date'].iloc[-1].date()}
   测试: {test_df['Date'].iloc[0].date()} → {test_df['Date'].iloc[-1].date()}
   总用时: {total_time / 3600:.2f} 小时 ({total_time:.0f} 秒)

🏗️ 模型配置:
   特征数量: {len(feature_columns)}  |  历史天数: {LOOKBACK}
   TCN 架构: 2×TemporalBlock(32 filters) + Dense(16) + Dense(1)

📊 第一阶段:
   训练次数: {N_FIRST_STAGE}  |  验证 MSE 范围: {min(r['val_MSE'] for r in first_stage_results):.4f} - {max(r['val_MSE'] for r in first_stage_results):.4f}
   平均训练时间: {np.mean([r['time'] for r in first_stage_results]):.1f}s/模型

📈 第二阶段（冠军模型）:
   种子: {best_test_result['seed']}  |  第一阶段排名: #{best_test_result['top_rank']}
   验证 MSE:  {best_test_result['val_MSE']:.4f}
   测试 MSE:  {best_test_result['test_MSE']:.4f}
   测试 RMSE: {best_test_result['test_RMSE']:.4f}
   测试 MAE:  {best_test_result['test_MAE']:.4f}
   测试 MAPE: {best_test_result['test_MAPE']:.2f}%
   测试 R²:   {best_test_result['test_R2']:.4f}
   均误差:    ${errors.mean():.2f}  |  误差标准差: ${errors.std():.2f}
   方向准确率: {direction_acc:.1f}%

📁 输出文件 (results/):
   stage1_all_runs.csv              - 第一阶段全部结果
   stage1_top{N_TOP_MODELS}_models.csv           - Top {N_TOP_MODELS} 模型信息
   stage2_top_test_results.csv      - 第二阶段测试结果
   final_best_predictions.csv       - 冠军模型预测值
   seed_optimization_history.csv    - 种子搜索历史
   model_performance_summary.png    - 综合性能图表
   error_analysis.png               - 误差分析图表
   seed_optimization_history.png    - 种子优化历史图表
   training_log.txt                 - 完整训练日志（本文件）

{'=' * 80}
"""
    print(report)

    # 单独保存一份纯文字报告
    with open(os.path.join(RESULTS_DIR, 'final_summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   ✅ 保存: {RESULTS_DIR}/final_summary_report.txt")

    print("\n🎉 全部完成！结果已保存至 results/ 文件夹。")

    logger.close()
    sys.stdout = logger.terminal   # 恢复正常 stdout


# ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
