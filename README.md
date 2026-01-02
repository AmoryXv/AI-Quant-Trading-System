# A-Share AI Quant System (LSTM + C++ Hybrid)

基于深度学习 (LSTM) 与 Optuna 自适应优化的 A 股量化择时系统。
An AI-driven quantitative trading system for A-Share market featuring LSTM predictions, C++ signal acceleration, and Walk-Forward backtesting.

## 🚀 Key Features (核心特性)

* **Hybrid Architecture**: Python (PyTorch) 负责模型训练与复杂风控，C++ (DLL) 负责毫秒级信号初筛，实现计算加速。
* **Deep Learning Alpha**: 使用 LSTM 网络提取非线性时序因子，经过 **Optuna** 全局参数寻优，测试集 Rank IC 达到 **0.08+**。
* **Robust Backtesting**: 构建了 **Walk-Forward (滚动时间窗)** 回测框架，杜绝未来函数 (Look-ahead Bias)。
* **Risk Management**: 集成 MA 趋势跟踪与大盘情绪 (Market Sentiment) 双重熔断机制，在 2023 年极端行情下实现 **16.76% 的纯 Alpha 超额收益**。
* **Visualization**: 集成 Streamlit 交互式看板，支持实盘信号监控与因子分析。

## 🛠️ Tech Stack (技术栈)

* **Core**: Python 3.9, C++17
* **ML/DL**: PyTorch (CUDA), XGBoost, Scikit-learn
* **Optimization**: Optuna (Bayesian Optimization)
* **Data/Backtest**: Pandas, Numpy, Joblib
* **Visualization**: Streamlit, Matplotlib, Plotly

## 📊 Performance (回测表现)

*Test Period: 2023/08 - 2024/01 (Walk-Forward Analysis)*

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Rank IC** | **0.0838** | Top-tier prediction quality |
| **Hedged Alpha** | **+16.76%** | Pure excess return vs Benchmark |
| **Long-Only Return**| **+0.96%** | Positive return in bear market (-15% drop) |
| **Win Rate** | High | Strict filtering (Active Days < 5%) |

## 📂 Project Structure

```text
Quant_System/
├── src/
│   ├── model_lstm.py       # LSTM Network & Sklearn-style Wrapper
│   ├── backtest_engine.py  # Walk-Forward Backtest Engine (Main)
│   ├── optimize.py         # Optuna Hyperparameter Tuning
│   ├── dashboard.py        # Streamlit Dashboard
│   ├── strategy.cpp        # C++ Signal Generation Source
│   └── build_cpp.py        # C++ Compilation Script
├── requirements.txt        # Dependencies
└── README.md