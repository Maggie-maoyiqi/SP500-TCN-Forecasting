import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 非交互后端，避免 macOS GUI 问题
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# 图表 1：综合性能总览（3×3 子图）
# ──────────────────────────────────────────────
def create_performance_chart(test_df, y_test, best_test_result,
                              first_stage_results, second_stage_results,
                              optimizer_summary, feature_columns,
                              lookback, n_first_stage, n_top_models,
                              results_dir):
    val_mses      = [r['val_MSE']  for r in first_stage_results]
    test_mses_top = [r['test_MSE'] for r in second_stage_results]
    val_mses_top  = [r['val_MSE']  for r in second_stage_results]
    top_ranks     = [r['top_rank'] for r in second_stage_results]
    champion_pred = best_test_result['predictions']
    champion_err  = champion_pred - y_test

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('S&P 500 TCN模型两阶段选择结果分析',
                 fontsize=18, fontweight='bold', y=0.98)

    # 子图1：预测 vs 实际
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(test_df['Date'], y_test, 'k-', label='实际价格', linewidth=2, alpha=0.8)
    ax1.plot(test_df['Date'], champion_pred, 'r--',
             label=f'预测 (R²={best_test_result["test_R2"]:.3f})', linewidth=2, alpha=0.8)
    ax1.set_title('冠军模型：预测 vs 实际', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期'); ax1.set_ylabel('价格 ($)')
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 子图2：第一阶段验证 MSE 分布
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(val_mses, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=best_test_result['val_MSE'], color='red', linestyle='--',
                linewidth=2, label=f'冠军: {best_test_result["val_MSE"]:.4f}')
    ax2.set_title(f'第一阶段验证MSE分布 ({n_first_stage}个模型)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('验证MSE'); ax2.set_ylabel('频次')
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    # 子图3：Top 模型测试 MSE 对比
    ax3 = plt.subplot(3, 3, 3)
    colors = ['red' if mse == best_test_result['test_MSE'] else 'gray'
              for mse in test_mses_top]
    bars = ax3.bar(top_ranks, test_mses_top, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title(f'Top {n_top_models} 模型：测试MSE对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('第一阶段排名'); ax3.set_ylabel('测试MSE (越低越好)')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, mse in zip(bars, test_mses_top):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{mse:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图4：验证 vs 测试 MSE 散点
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(val_mses_top, test_mses_top, s=150, c=top_ranks,
                cmap='viridis', edgecolor='black', alpha=0.8)
    for r in second_stage_results:
        color = 'red' if r['test_MSE'] == best_test_result['test_MSE'] else 'black'
        ax4.annotate(f'#{r["top_rank"]}', (r['val_MSE'], r['test_MSE']),
                     fontsize=11, ha='center', fontweight='bold', color=color)
    ax4.set_title('验证集MSE vs 测试集MSE', fontsize=14, fontweight='bold')
    ax4.set_xlabel('验证集MSE'); ax4.set_ylabel('测试集MSE')
    ax4.grid(True, alpha=0.3)

    # 子图5：误差时间序列
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(test_df['Date'], champion_err, color='purple', linewidth=1.5)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.fill_between(test_df['Date'], champion_err, 0, alpha=0.3, color='purple')
    ax5.set_title('误差时间序列', fontsize=14, fontweight='bold')
    ax5.set_xlabel('日期'); ax5.set_ylabel('误差 ($)')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 子图6：误差分布
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(champion_err, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.set_title(f'误差分布 (均值={champion_err.mean():.3f})',
                  fontsize=14, fontweight='bold')
    ax6.set_xlabel('误差 ($)'); ax6.set_ylabel('频次')
    ax6.grid(True, alpha=0.3)

    # 子图7：实际 vs 预测散点
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(y_test, champion_pred, alpha=0.6, c=range(len(y_test)),
                cmap='viridis', s=30, edgecolor='black', linewidth=0.5)
    mn = min(y_test.min(), champion_pred.min())
    mx = max(y_test.max(), champion_pred.max())
    ax7.plot([mn, mx], [mn, mx], 'r--', linewidth=3, label='完美预测线', alpha=0.8)
    ax7.set_title(f'实际 vs 预测 (R²={best_test_result["test_R2"]:.4f})',
                  fontsize=14, fontweight='bold')
    ax7.set_xlabel('实际价格 ($)'); ax7.set_ylabel('预测价格 ($)')
    ax7.legend(); ax7.grid(True, alpha=0.3)

    # 子图8：种子优化历史  ← BUG FIXED: 保证 x/y 长度一致
    ax8 = plt.subplot(3, 3, 8)
    n_show     = min(100, len(optimizer_summary['loss_history']))
    iterations = list(range(1, n_show + 1))
    losses     = optimizer_summary['loss_history'][:n_show]
    ax8.plot(iterations, losses, 'o-', color='green', linewidth=2, markersize=4)
    ax8.axhline(y=optimizer_summary['best_loss'], color='red', linestyle='--',
                linewidth=2, alpha=0.5,
                label=f'最佳: {optimizer_summary["best_loss"]:.4f}')
    ax8.set_title(f'种子优化历史（前 {n_show} 次）', fontsize=14, fontweight='bold')
    ax8.set_xlabel('迭代次数'); ax8.set_ylabel('验证MSE')
    ax8.legend(); ax8.grid(True, alpha=0.3)

    # 子图9：文字总结
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = (
        f"🏆 冠军模型总结\n"
        f"种子: {best_test_result['seed']}\n"
        f"第一阶段排名: #{best_test_result['top_rank']}\n\n"
        f"📊 验证集:\n"
        f"  MSE:  {best_test_result['val_MSE']:.4f}\n"
        f"  RMSE: {best_test_result['val_RMSE']:.4f}\n"
        f"  MAE:  {best_test_result['val_MAE']:.4f}\n\n"
        f"📈 测试集:\n"
        f"  MSE:  {best_test_result['test_MSE']:.4f}\n"
        f"  RMSE: {best_test_result['test_RMSE']:.4f}\n"
        f"  MAE:  {best_test_result['test_MAE']:.4f}\n"
        f"  MAPE: {best_test_result['test_MAPE']:.2f}%\n"
        f"  R²:   {best_test_result['test_R2']:.4f}\n\n"
        f"🔧 配置:\n"
        f"  特征数: {len(feature_columns)}\n"
        f"  历史天数: {lookback}\n"
        f"  训练次数: {n_first_stage}"
    )
    ax9.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(results_dir, 'model_performance_summary.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: {path}")
    return champion_err


# ──────────────────────────────────────────────
# 图表 2：误差分析
# ──────────────────────────────────────────────
def create_error_analysis_chart(test_df, champion_errors, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(test_df['Date'], np.cumsum(champion_errors), 'b-', linewidth=2)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_title('累积误差（应围绕0波动）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期'); ax1.set_ylabel('累积误差 ($)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    window      = 30
    rolling_mae = pd.Series(np.abs(champion_errors)).rolling(window).mean()
    rolling_std = pd.Series(champion_errors).rolling(window).std()
    ax2.plot(test_df['Date'].iloc[window - 1:], rolling_mae.iloc[window - 1:],
             'g-', label=f'{window}日平均绝对误差', linewidth=2)
    ax2.plot(test_df['Date'].iloc[window - 1:], rolling_std.iloc[window - 1:],
             'r-', label=f'{window}日误差标准差', linewidth=2)
    ax2.set_title(f'{window}日滚动误差统计', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期'); ax2.set_ylabel('误差 ($)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    path = os.path.join(results_dir, 'error_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: {path}")


# ──────────────────────────────────────────────
# 图表 3：种子搜索历史
# ──────────────────────────────────────────────
def create_seed_history_chart(optimizer_summary, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(optimizer_summary['seed_history'], optimizer_summary['loss_history'],
             'o-', alpha=0.5, markersize=3)
    ax1.scatter([optimizer_summary['best_seed']], [optimizer_summary['best_loss']],
                color='red', s=200, marker='*', label='最佳种子', zorder=5)
    ax1.set_title('完整种子搜索历史', fontsize=14, fontweight='bold')
    ax1.set_xlabel('种子值'); ax1.set_ylabel('验证MSE')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.hist(optimizer_summary['loss_history'], bins=30,
             color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(x=optimizer_summary['best_loss'], color='red', linestyle='--',
                linewidth=2, label=f'最佳: {optimizer_summary["best_loss"]:.4f}')
    ax2.set_title('验证损失分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('验证MSE'); ax2.set_ylabel('频次')
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(results_dir, 'seed_optimization_history.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存: {path}")
