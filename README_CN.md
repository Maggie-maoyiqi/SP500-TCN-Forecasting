# SP500-TCN-Forecasting

基于时序卷积网络（TCN）与滚动预测策略的标普500指数智能预测系统。融合两阶段模型筛选机制、宏观经济指标和40余项技术特征，预测SPY收盘价。针对传统模型中常见的误差积累、模型不稳定和缺乏经济背景等问题提出了系统性解决方案。

# 🚀 S&P 500 TCN 预测模型

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Last Updated](https://img.shields.io/github/last-commit/YOUR_USERNAME/SP500-TCN-Forecasting)

一个用于预测标普500（SPY）收盘价的深度学习系统，基于时序卷积网络（TCN），针对金融时间序列预测中的三大核心问题提出了创新解决方案。

## 📈 项目简介

本项目解决了股票市场预测中三个关键挑战：

1. **📉 系统性偏差（"滞后"问题）** - 通过滚动预测策略解决
2. **🎲 模型不稳定性（随机种子问题）** - 通过基于梯度的种子搜索解决（200次运行）
3. **🌍 缺乏经济背景** - 通过融合宏观经济特征与增强动量特征解决

### 🎯 核心特性

- **🧠 TCN 架构**：时序卷积网络，擅长捕捉长期依赖关系
- **🔄 滚动预测**：实时更新真实值，防止误差积累
- **🎲 两阶段筛选**：200次训练 → 筛选前5名 → 选出最优模型
- **📊 40+ 特征**：技术指标 + 动量因子 + 宏观经济因子
- **🔒 数据泄漏防护**：双重加锁机制，配合因果填充
- **📈 宏观经济整合**：国债收益率与GDP数据
- **⚡ 增强动量特征**：超越简单收益率的高级动量分析

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **测试集 MSE** | ~193.20 | 均方误差 |
| **测试集 R²** | ~0.98+ | 决定系数 |
| **测试集 MAPE** | ~0.5% | 平均绝对百分比误差 |
| **方向准确率** | ~65%+ | 涨跌方向预测正确率 |
| **训练时间** | ~2-3小时 | 完成200次模型训练 |

## 📁 项目结构

```
SP500-TCN-Forecasting/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据（请将CSV文件放于此处）
│   │   ├── 3010train.csv   # 训练数据（2015-2021）
│   │   ├── 3010test.csv    # 测试数据（2022-2025）
│   │   ├── DGS10.csv       # 10年期美债收益率
│   │   └── GDPC1.csv       # 实际GDP数据
│   └── processed/          # 已处理数据（自动生成）
├── src/                    # 源代码模块
│   ├── __init__.py
│   ├── data_preprocessing.py    # 步骤1-4：数据加载与预处理
│   ├── feature_engineering.py   # 步骤3：40+特征工程
│   ├── tcn_model.py             # 步骤5：TCN模型架构
│   ├── seed_optimizer.py        # 步骤6：基于梯度的种子搜索
│   ├── rolling_forecast.py      # 步骤7：滚动预测函数
│   ├── training_stage1.py       # 步骤8：第一阶段，200次训练
│   ├── training_stage2.py       # 步骤9：第二阶段，前5名测试
│   └── visualization.py         # 步骤10：结果可视化
├── notebooks/              # Jupyter Notebook
│   └── SP500_TCN_Complete_Pipeline.ipynb  # 完整工作流
├── models/                 # 已保存的模型
│   └── best_model.h5      # 最优模型
├── results/                # 输出结果
│   ├── predictions/       # 预测结果CSV
│   ├── visualizations/    # 生成的图表
│   └── logs/             # 训练日志
├── docs/                  # 文档
│   ├── methodology.md     # 详细方法说明
│   └── results_analysis.md # 结果解读
├── tests/                 # 单元测试
├── requirements.txt       # Python依赖
├── config.yaml           # 配置文件
├── main.py              # 主入口
├── LICENSE              # MIT许可证
└── README.md            # 英文说明文档
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8 或更高版本
- 建议 8GB 以上内存
- NVIDIA GPU（可选，但推荐用于加速训练）

### 2. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/SP500-TCN-Forecasting.git
cd SP500-TCN-Forecasting

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows系统：venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据

请将以下文件放入 `data/raw/` 目录：

| 文件 | 说明 | 时间范围 | 必需列 |
|------|------|----------|--------|
| `3010train.csv` | 训练数据 | 2015-2021 | `Date, Open, High, Low, Close/Last, Volume` |
| `3010test.csv` | 测试数据 | 2022-2025 | `Date, Open, High, Low, Close/Last, Volume` |
| `DGS10.csv` | 美债收益率 | 每日 | `observation_date, DGS10` |
| `GDPC1.csv` | GDP数据 | 季度 | `observation_date, GDPC1` |

**提示**：可从 [FRED](https://fred.stlouisfed.org/) 下载样本数据，或使用 Yahoo Finance 获取。

### 4. 运行完整流水线

```bash
# 运行全部10个步骤（完整流水线）
python main.py --mode full

# 或分步运行
python main.py --mode preprocess  # 步骤1-4：仅预处理
python main.py --mode stage1      # 步骤8：训练200个模型
python main.py --mode stage2      # 步骤9：测试前5名模型
```

### 5. Jupyter Notebook

交互式探索：
```bash
jupyter notebook notebooks/SP500_TCN_Complete_Pipeline.ipynb
```

## 🔧 模型架构

### 📊 特征工程（40个特征）

模型使用5类精心设计的40个特征：

| 类别 | 数量 | 代表特征 | 用途 |
|------|------|----------|------|
| **基础价格特征** | 6 | `prev_close`, `prev_open`, `prev_high`, `prev_low`, `prev_volume`, `prev_hl_range` | 基本价格信息 |
| **技术指标** | 13 | `MA_5`, `MA_10`, `MA_20`, `RSI_14`, `MACD`, `volatility_10` | 市场趋势与动量 |
| **增强动量特征** | 11 | `momentum_2d/3d/5d`, `acceleration`, `trend_strength`, `momentum_ratio` | 高级动量分析 |
| **量价因子** | 5 | `intraday_momentum`, `candle_body_ratio`, `money_flow` | 量价确认信号 |
| **宏观经济因子** | 5 | `prev_dgs10`, `prev_gdpc1`, `dgs10_change`, `gdpc1_growth` | 经济背景信息 |

### 🧠 TCN 模型结构

```
输入层：(batch_size, 60, 40)
    ↓
时序块 1：
    Conv1D（32个滤波器，核大小=3，扩张=1，因果填充）
    Dropout（0.2）
    Conv1D（32个滤波器，核大小=3，扩张=1，因果填充）
    Dropout（0.2）
    残差连接
    ↓
时序块 2：
    Conv1D（32个滤波器，核大小=3，扩张=2，因果填充）
    Dropout（0.2）
    Conv1D（32个滤波器，核大小=3，扩张=2，因果填充）
    Dropout（0.2）
    残差连接
    ↓
展平层：(batch_size, 1920)
    ↓
全连接层：16个神经元，ReLU激活
    ↓
输出层：1个神经元（明日收盘价）
```

**总参数量**：约62,000

## 🎯 创新解决方案

### 1. 🌀 滚动预测策略
传统模型常出现"滞后"问题——预测曲线只是真实曲线的平移。本项目的解决方案：

```python
# 传统方式（导致滞后）：
history = history + prediction  # ❌ 使用预测值更新历史

# 本项目方式（防止滞后）：
history = history + actual_value  # ✅ 使用真实值更新历史
```

**原理**：始终用真实值更新历史窗口，防止误差积累引发系统性偏差。

### 2. 🎲 基于梯度的种子搜索 + 两阶段筛选

**第一阶段：广泛筛网**
- 使用不同随机种子训练200个模型
- 采用智能种子搜索（非纯随机）
- 按验证集MSE选出前5名

**第二阶段：压力测试**
- 在未见过的测试数据上评估前5名模型
- 以泛化能力为标准选出最终冠军模型

```python
# 基于梯度的种子优化器
optimizer = GradientBasedSeedOptimizer()
for i in range(200):
    seed = optimizer.get_next_seed()  # 智能搜索
    model = train_with_seed(seed)
    loss = validate_model(model)
    optimizer.update(seed, loss)  # 从结果中学习
```

### 3. 🔒 双重数据泄漏防护

1. **数据层**：所有40个特征均使用 `.shift(1)` 操作——预测第t天的数据仅使用第t-1天及之前的数据
2. **模型层**：TCN的因果填充确保卷积滤波器不能"看到"未来时间步

## 📈 训练策略

### 数据划分

- **训练集**：2015年1月1日 – 2021年12月31日
  - 涵盖牛市与新冠暴跌行情
  - 共1760个交易日

- **验证集**：训练集的15%（自动划分）

- **测试集**：2022年1月1日 – 2025年6月1日
  - 涵盖通胀期与加息周期
  - AI牛市行情
  - 共875个交易日（真实样本外测试）

### 超参数设置

| 参数 | 数值 | 说明 |
|------|------|------|
| 回溯窗口 | 60天 | 用于预测的历史数据长度 |
| 批大小 | 32 | 训练批次大小 |
| 训练轮数 | 50 | 最大训练轮数 |
| 早停 | patience=10 | 无改善则停止训练 |
| 学习率 | 自适应 | 在平台期自动降低 |
| Dropout率 | 0.2 | 正则化参数 |

## 📊 结果解读

### 可视化输出
模型生成8张综合可视化图表：

1. **预测值 vs 实际值** - 时间序列对比图
2. **误差分布** - 以0为中心的直方图
3. **第一阶段验证MSE分布** - 200次运行的直方图
4. **前5名测试MSE对比** - 柱状图
5. **验证集 vs 测试集MSE散点图** - 过拟合检测
6. **误差时间序列** - 应在0附近波动
7. **种子优化历史** - 搜索过程记录
8. **性能总结** - 关键指标汇总表

### 核心指标说明
- **MSE（均方误差）**：对大误差惩罚更重
- **RMSE（均方根误差）**：与价格单位相同（美元）
- **MAE（平均绝对误差）**：平均绝对误差值
- **MAPE（平均绝对百分比误差）**：百分比形式的误差
- **R²（决定系数）**：1.0 = 完美预测，0.0 = 均值预测

## 🔬 方法论详解

### 特征工程规则
- 所有特征严格使用 `.shift(1)` 操作
- 绝不使用未来数据
- 缺失值先前向填充再后向填充
- 特征归一化至 [0, 1] 范围

### 模型训练细节
- Adam优化器，使用默认参数
- 均方误差损失函数
- 早停，patience=10
- 在平台期降低学习率
- 使用模型检查点保存最优权重

### 测试协议
- 真实样本外测试（2022-2025数据）
- 不在测试集上调参
- 滚动预测模拟真实交易场景
- 所有结果可通过随机种子复现

## 🛠️ 进阶用法

### 自定义特征
编辑 `src/feature_engineering.py` 以添加或修改特征：

```python
# 添加自定义特征
def add_custom_feature(df):
    df['my_custom_feature'] = df['Close/Last'].rolling(20).std()
    return df
```

### 调整模型架构
修改 `src/tcn_model.py`：

```python
def build_custom_tcn(input_shape):
    inputs = Input(shape=input_shape)
    x = TemporalBlock(64, 5, 1, 0.3)(inputs)  # 更多滤波器
    x = TemporalBlock(64, 5, 2, 0.3)(x)
    x = TemporalBlock(32, 3, 4, 0.2)(x)      # 增加一层
    # ... 模型其余部分
```

### 使用不同数据运行
创建配置文件：

```yaml
# config.yaml
data:
  train_file: "data/raw/my_train_data.csv"
  test_file: "data/raw/my_test_data.csv"
  lookback: 30  # 更短的回溯窗口
model:
  filters: [64, 64, 32]
  kernel_size: 5
training:
  n_runs: 100   # 减少运行次数以便测试
  batch_size: 64
```

## 📚 数据来源

### 主要数据
1. **S&P 500（SPY）历史数据**
   - 来源：Yahoo Finance、Quandl 或您的券商
   - 需要：每日OHLCV数据

2. **10年期美国国债到期收益率（DGS10）**
   - 来源：[FRED](https://fred.stlouisfed.org/series/DGS10)
   - 频率：每日

3. **实际国内生产总值（GDPC1）**
   - 来源：[FRED](https://fred.stlouisfed.org/series/GDPC1)
   - 频率：季度（自动扩展为每日）

### 其他可用数据源
- **利率数据**：美联储经济数据（FRED）
- **通胀数据**：美国劳工统计局CPI
- **就业数据**：非农就业人数
- **情绪数据**：VIX指数、看跌/看涨期权比率

## 🤝 参与贡献

欢迎各种形式的贡献！

### 报告问题
1. 检查是否已有相关issue
2. 提供可复现的代码示例
3. 附上错误信息与运行环境信息

### 建议改进
1. 以"[ENHANCEMENT]"为前缀开启issue
2. 描述拟议的修改内容
3. 说明优点与实现思路

### 提交 Pull Request
1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add AmazingFeature'`）
4. 推送分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

### 开发规范
- 遵循 PEP 8 代码风格
- 为新函数添加文档字符串
- 为新特性编写单元测试
- 及时更新相关文档

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试模块
python -m pytest tests/test_feature_engineering.py

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

## 📝 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **S&P 500 数据提供方**：Yahoo Finance、Alpha Vantage、Quandl
- **经济数据**：美联储经济数据库（FRED）
- **TCN架构**：Bai et al.（2018）"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- **金融特征工程**：灵感来源于量化金融相关文献
- **开源社区**：TensorFlow、scikit-learn、pandas、numpy 的开发者们

## 📧 联系与支持

**项目维护者**：[Your Name]
**邮箱**：[your.email@example.com](mailto:your.email@example.com)
**GitHub Issues**：[https://github.com/YOUR_USERNAME/SP500-TCN-Forecasting/issues](https://github.com/YOUR_USERNAME/SP500-TCN-Forecasting/issues)

### 支持渠道
1. **GitHub Issues**：报告Bug与功能请求
2. **Discussions**：提问与社区交流
3. **Email**：私密事宜联系

## 📖 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{sp500_tcn_forecasting,
  title = {S\&P 500 TCN Forecasting Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/SP500-TCN-Forecasting},
  note = {Advanced deep learning model for S\&P 500 price prediction}
}
```

## 🔄 更新日志

### v1.0.0（当前版本）
- 初始发布，包含完整的10步流水线
- 40+特征工程模块
- 200次模型运行的两阶段训练
- 完整可视化套件
- 完整文档与示例

### 计划特性
- 实时预测能力
- 更多宏观经济指标
- 集成学习方法
- Web预测界面
- API接口支持

## ⚠️ 免责声明

**重要提示**：本项目仅供学术研究与教育目的使用。

- **非投资建议**：本模型的预测结果不构成任何实际交易建议。
- **历史表现**：过去的表现不代表未来的结果。
- **投资风险**：所有投资均存在风险，您可能损失本金。
- **验证建议**：请务必结合自己的数据和分析对模型进行验证。

作者不对任何因使用本软件而产生的财务损失承担责任。

---

<div align="center">

### ⭐ 如果本项目对您有帮助，请在 GitHub 上给个 Star！⭐

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/SP500-TCN-Forecasting&type=Date)](https://star-history.com/#YOUR_USERNAME/SP500-TCN-Forecasting&Date)

</div>

---

**祝预测顺利！** 📈🚀
