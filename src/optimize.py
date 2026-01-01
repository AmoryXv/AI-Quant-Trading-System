# src/optimize.py
# AI Hyperparameter Tuning using Optuna
# Optimizing both Model Architecture and Strategy Logic

import optuna
import pandas as pd
import numpy as np
import os
import ctypes
import torch
import warnings
from model_lstm import LSTMPredictor
from xgboost import XGBRegressor

# Configuration
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# 1. 准备环境与数据
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data_processed')
dll_path = os.path.join(project_root, 'strategy.dll')

# 加载 C++ 库
if not os.path.exists(dll_path):
    raise FileNotFoundError("Strategy DLL not found.")
lib = ctypes.CDLL(dll_path)
lib.generate_signals.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
    ctypes.c_int, ctypes.c_double
]
lib.generate_signals.restype = None

def load_data():
    file_list = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    all_data = []
    for f in file_list:
        df = pd.read_parquet(os.path.join(data_path, f))
        all_data.append(df)
    dataset = pd.concat(all_data, ignore_index=True)
    return dataset.replace([np.inf, -np.inf], np.nan).dropna().sort_values('trade_date')

# 预加载数据，避免每次 trial 都读盘
FULL_DF = load_data()
FEATURE_COLS = ['ROC_5', 'ROC_20', 'Vol_20', 'RSI', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST']
TARGET_COL = 'Future_Ret_5'

# 划分验证集 (Validation Set)
# 注意：我们用 2023-01 到 2023-09 做训练，用 2023-10 到 2023-12 做验证优化
# 这样能保证我们选出来的参数在"困难模式"（2023Q4）下也是有效的
SPLIT_DATE = pd.to_datetime('2023-10-01')
TRAIN_DF = FULL_DF[FULL_DF['trade_date'] < SPLIT_DATE].copy()
VAL_DF = FULL_DF[FULL_DF['trade_date'] >= SPLIT_DATE].copy()

print(f"[System] Data Loaded. Train: {len(TRAIN_DF)} | Val: {len(VAL_DF)}")

# ==========================================
# 2. 定义优化目标函数 (Objective Function)
# ==========================================
def objective(trial):
    # -----------------------------------
    # A. 采样超参数 (Hyperparameters)
    # -----------------------------------
    # 1. 模型参数
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 10, 30)
    
    # 2. 策略风控参数
    ma_window = trial.suggest_int("ma_window", 10, 60, step=5) # 均线周期
    min_market_sentiment = trial.suggest_float("sentiment_threshold", -0.001, 0.002) # 大盘情绪门槛
    min_stock_pred = trial.suggest_float("stock_threshold", 0.001, 0.008) # 个股涨幅门槛
    top_k = trial.suggest_int("top_k", 1, 5) # 买前几名
    
    # -----------------------------------
    # B. 训练 LSTM 模型
    # -----------------------------------
    # 为了速度，batch_size 固定大一点
    model = LSTMPredictor(
        sequence_length=10,
        epochs=epochs,
        hidden_dim=hidden_dim,
        batch_size=2048, # 加速优化过程
        learning_rate=learning_rate
    )
    
    # 训练
    try:
        model.fit(TRAIN_DF, FEATURE_COLS, TARGET_COL)
        preds = model.predict(VAL_DF, FEATURE_COLS)
    except Exception as e:
        print(f"Trial failed: {e}")
        return -999 # 惩罚失败的尝试
        
    # -----------------------------------
    # C. 计算策略表现 (Backtest Logic)
    # -----------------------------------
    test_df = VAL_DF.copy()
    test_df['pred_lstm'] = preds
    
    # 1. 计算 Rank IC (作为基础分)
    temp_eval = test_df.dropna(subset=['pred_lstm', TARGET_COL])
    if len(temp_eval) == 0: return -999
    rank_ic = temp_eval[['pred_lstm', TARGET_COL]].corr(method='spearman').iloc[0,1]
    
    # 2. 策略逻辑应用
    # 计算 MA
    market_index = test_df.groupby('trade_date')['close'].mean()
    market_ma = market_index.rolling(window=ma_window).mean()
    test_df['market_price'] = test_df['trade_date'].map(market_index)
    test_df['market_ma'] = test_df['trade_date'].map(market_ma)
    
    # C++ Filter
    final_preds = test_df['pred_lstm'].fillna(-999).values
    c_signals = np.zeros(len(final_preds), dtype=np.float64)
    lib.generate_signals(
        final_preds.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
        c_signals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
        len(final_preds), 
        ctypes.c_double(0.0)
    )
    test_df['Signal_Raw'] = c_signals
    
    # Python Filter
    test_df['Rank'] = test_df.groupby('trade_date')['pred_lstm'].rank(ascending=False, method='first')
    market_sentiment = test_df.groupby('trade_date')['pred_lstm'].transform('mean')
    trend_condition = test_df['market_price'] > test_df['market_ma']
    
    condition = (
        (test_df['Signal_Raw'] == 1) & 
        (test_df['Rank'] <= top_k) & 
        (test_df['pred_lstm'] > min_stock_pred) & 
        (market_sentiment > min_market_sentiment) & 
        (trend_condition)
    )
    test_df['Final_Signal'] = np.where(condition, 1.0, 0.0)
    
    # 3. 计算收益指标
    # 引入交易成本
    daily_gross_ret = test_df[test_df['Final_Signal'] == 1.0].groupby('trade_date')[TARGET_COL].mean().fillna(0.0)
    benchmark_ret = test_df.groupby('trade_date')[TARGET_COL].mean().fillna(0.0)
    
    # 构造完整时间序列
    all_dates = test_df['trade_date'].unique()
    strategy_series = pd.Series(0.0, index=all_dates)
    strategy_series.update(daily_gross_ret)
    
    # 扣费 (万20)
    strategy_net = np.where(strategy_series != 0, strategy_series - 0.002, 0.0)
    
    # 计算超额收益 (Alpha Return)
    alpha_daily = (strategy_net - benchmark_ret) / 5
    cumulative_alpha = (1 + alpha_daily).cumprod()[-1] - 1
    
    # 计算夏普 (Alpha Sharpe)
    if np.std(alpha_daily) == 0:
        alpha_sharpe = 0
    else:
        alpha_sharpe = (np.mean(alpha_daily) / np.std(alpha_daily)) * np.sqrt(250)

    # -----------------------------------
    # D. 定义最终分数 (Optimization Goal)
    # -----------------------------------
    # 我们希望：IC 高，且 实盘 Alpha 收益高
    # 混合打分：Score = RankIC * 0.3 + AlphaReturn * 0.7
    # 这样更偏向于"能赚钱"的参数
    
    final_score = (rank_ic * 0.3) + (cumulative_alpha * 0.7)
    
    return final_score

# ==========================================
# 3. 运行优化
# ==========================================
if __name__ == "__main__":
    print("[System] Starting Optuna Optimization...")
    print("  - Target: Maximize (0.3 * RankIC + 0.7 * AlphaReturn)")
    print("  - Trials: 30")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30) # 跑30次实验，大概几分钟
    
    print("\n" + "="*50)
    print(" [Optimization Complete]")
    print("="*50)
    print(f" Best Score: {study.best_value:.4f}")
    print(" Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")
        
    print("\n[Action] Please update 'src/backtest_engine.py' with these parameters!")