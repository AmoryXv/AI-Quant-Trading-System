# AI-Driven Quant Trading System (A股量化选股系统)

## Project Overview (项目简介)
基于 XGBoost 与多因子模型（Multi-Factor Model）的 A 股量化选股系统。项目实现了从数据清洗、因子挖掘、机器学习建模到回测分析的全流程闭环。在 2023 Q4 的熊市环境下，策略跑赢沪深300基准指数约 5%，展现了显著的超额收益能力。

## Key Features (核心特性)
* **Data Engineering**: 封装 Baostock 接口，实现 300+ 股票的自动清洗与前复权处理 (Parquet 存储)。
* **Factor Mining**: 构建了动量 (ROC, RSI)、波动率 (Volatility)、趋势 (MACD) 等多维度因子库。
* **ML Modeling**: 采用 XGBoost 进行滚动窗口训练，RankIC 达到 0.06，有效捕捉市场短期规律。
* **Backtesting**: 实现了 Top-K 轮动策略回测引擎，支持 Python/C++ 混合调用 (Hybrid Architecture)。

## Tech Stack (技术栈)
* **Core**: Python 3.9, Pandas, NumPy
* **ML**: XGBoost, Scikit-Learn
* **Performance**: C++ (DLL via ctypes)
* **Data**: Baostock, Parquet

## Performance (回测表现)
* **Benchmark**: CSI 300 (沪深300)
* **Period**: 2023.10.01 - 2023.12.31
* **Strategy Return**: -1.75% (vs Benchmark ~ -8%)
* **Alpha**: +6.25%