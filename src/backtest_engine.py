# src/backtest_engine.py
# Final Production Version: Robust Architecture with Full Persistence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import warnings
import joblib  # [NEW] 用于保存归一化参数，工程必备
import torch
from xgboost import XGBRegressor
from model_lstm import LSTMPredictor

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class BacktestSystem:
    def __init__(self):
        # 自动定位项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_root, 'data_processed')
        self.dll_path = os.path.join(self.project_root, 'strategy.dll')
        
        # Link C++ Library
        if not os.path.exists(self.dll_path):
             raise FileNotFoundError(f"Critical: {self.dll_path} missing!")
        
        try:
            self.lib = ctypes.CDLL(self.dll_path)
            # Explicitly define argument types for C++ function
            self.lib.generate_signals.argtypes = [
                ctypes.POINTER(ctypes.c_double), # input array
                ctypes.POINTER(ctypes.c_double), # output array
                ctypes.c_int,                    # length
                ctypes.c_double                  # threshold
            ]
            self.lib.generate_signals.restype = None
            
        except Exception as e:
            raise RuntimeError(f"DLL Load Error: {e}")

    def load_data(self):
        print("[System] Loading Parquet Data...")
        file_list = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        all_data = []
        for f in file_list:
            df = pd.read_parquet(os.path.join(self.data_path, f))
            all_data.append(df)
        if not all_data: raise ValueError("No data found!")
        dataset = pd.concat(all_data, ignore_index=True)
        return dataset.replace([np.inf, -np.inf], np.nan).dropna().sort_values('trade_date')

    def run_backtest(self):
        df = self.load_data()
        
        # 1. Data Splitting
        split_date = pd.to_datetime('2023-10-01')
        train_df = df[df['trade_date'] < split_date].copy()
        test_df = df[df['trade_date'] >= split_date].copy()
        
        if len(test_df) == 0:
            raise ValueError("Test set is empty! Check split_date.")

        feature_cols = ['ROC_5', 'ROC_20', 'Vol_20', 'RSI', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST']
        target_col = 'Future_Ret_5'
        
        print(f"[Data] Train: {len(train_df)} | Test: {len(test_df)}")
        
        # ==========================================
        # 2. Benchmark Model: XGBoost
        # ==========================================
        print("\n--- Training Benchmark: XGBoost ---")
        xgb_model = XGBRegressor(max_depth=4, n_estimators=100, n_jobs=-1, random_state=42)
        xgb_model.fit(train_df[feature_cols], train_df[target_col])
        test_df['pred_xgb'] = xgb_model.predict(test_df[feature_cols])
        
        # ==========================================
        # 3. Core Model: LSTM (High Precision Mode)
        # ==========================================
        print("\n--- Training Core: LSTM (PyTorch) ---")
        lstm_model = LSTMPredictor(
            sequence_length=10, 
            epochs=20, 
            hidden_dim=64, 
            batch_size=1024,
            learning_rate=0.001
        )
        lstm_model.fit(train_df, feature_cols, target_col)
        test_df['pred_lstm'] = lstm_model.predict(test_df, feature_cols)
        
        # IC Evaluation
        eval_df = test_df.dropna(subset=['pred_lstm', target_col])
        ic_lstm = eval_df[['pred_lstm', target_col]].corr(method='spearman').iloc[0,1]
        
        print("\n" + "="*45)
        print(f" LSTM Rank IC: {ic_lstm:.4f} (Prediction Quality)")
        print("="*45 + "\n")

        # ==========================================
        # 4. Strategy Logic (Strict Trend Filter)
        # ==========================================
        
        # --- 计算技术指标 ---
        market_index = test_df.groupby('trade_date')['close'].mean()
        market_ma20 = market_index.rolling(window=20).mean()
        
        test_df['market_price'] = test_df['trade_date'].map(market_index)
        test_df['market_ma20'] = test_df['trade_date'].map(market_ma20)
        
        # --- C++ Pre-filter ---
        final_preds = test_df['pred_lstm'].fillna(-999).values
        c_signals = np.zeros(len(final_preds), dtype=np.float64)
        
        self.lib.generate_signals(
            final_preds.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            c_signals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            len(final_preds), 
            ctypes.c_double(0.0) 
        )
        test_df['Signal_Raw'] = c_signals 

        # --- Python Selection Logic ---
        test_df['Rank'] = test_df.groupby('trade_date')['pred_lstm'].rank(ascending=False, method='first')
        
        # 计算大盘情绪
        market_sentiment = test_df.groupby('trade_date')['pred_lstm'].transform('mean')

        # 终极风控条件
        trend_condition = test_df['market_price'] > test_df['market_ma20']
        
        # [Optimization] 微调参数以消除假突破回撤
        min_market_sentiment = 0.0005 # 稍微提高门槛，更加稳健
        
        condition = (
            (test_df['Signal_Raw'] == 1) & 
            (test_df['Rank'] <= 3) &              # Top 3
            (test_df['pred_lstm'] > 0.005) &      # 个股 > 0.5%
            (market_sentiment > min_market_sentiment) & # 大盘情绪看涨
            (trend_condition)                     # 且站上 20 日线
        )
        
        test_df['Final_Signal'] = np.where(condition, 1.0, 0.0)
        
        trade_days = test_df[test_df['Final_Signal'] == 1.0]['trade_date'].nunique()
        print(f">> Strategy Active Days: {trade_days} / {test_df['trade_date'].nunique()} (Target: Low)")

        # ==========================================
        # 5. Performance & Alpha Visualization
        # ==========================================
        
        # 1. 提取所有交易日索引
        all_dates = np.sort(test_df['trade_date'].unique())
        
        # 2. 计算收益
        daily_ret = test_df[test_df['Final_Signal'] == 1.0].groupby('trade_date')['Future_Ret_5'].mean()
        benchmark_ret = test_df.groupby('trade_date')['Future_Ret_5'].mean()
        
        # 3. 对齐数据
        strategy_df = pd.DataFrame(index=all_dates)
        strategy_df['Benchmark'] = benchmark_ret
        strategy_df['Strategy'] = daily_ret
        
        # 4. 填充空仓日
        strategy_df['Strategy'] = strategy_df['Strategy'].fillna(0.0)
        strategy_df['Benchmark'] = strategy_df['Benchmark'].fillna(0.0)
        
        # 5. 净值计算
        strategy_df['Equity_LongOnly'] = (1 + strategy_df['Strategy']/5).cumprod()
        strategy_df['Equity_Bench'] = (1 + strategy_df['Benchmark']/5).cumprod()
        
        # 6. Alpha 计算 (Hedged)
        strategy_df['Alpha_Ret'] = (strategy_df['Strategy'] - strategy_df['Benchmark']) / 5
        strategy_df['Equity_Alpha'] = (1 + strategy_df['Alpha_Ret']).cumprod()

        total_ret = strategy_df['Equity_LongOnly'].iloc[-1] - 1
        alpha_ret = strategy_df['Equity_Alpha'].iloc[-1] - 1
        
        print("\n" + "="*45)
        print(f" [Final Performance]")
        print(f" Long-Only Return: {total_ret*100:6.2f}%  (Real PnL)")
        print(f" Hedged Alpha:     {alpha_ret*100:6.2f}%  (Model Power)")
        print("="*45 + "\n")
        
        # ==========================================
        # 6. Model Persistence (工程落地关键步骤)
        # ==========================================
        model_dir = os.path.join(self.project_root, 'models')
        os.makedirs(model_dir, exist_ok=True)
        print(f"[System] Saving artifacts to {model_dir}...")

        # A. 保存 PyTorch 权重
        torch.save(lstm_model.model.state_dict(), os.path.join(model_dir, 'lstm_best.pth'))
        
        # B. [关键] 保存归一化器 (Joblib)
        joblib.dump(lstm_model.scaler_X, os.path.join(model_dir, 'scaler_x.pkl'))
        joblib.dump(lstm_model.scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))
        
        # C. 保存 XGBoost
        xgb_model.save_model(os.path.join(model_dir, 'xgb_benchmark.json'))
        
        print(f"  - All models and scalers saved successfully.")

        # ==========================================
        # 7. Plotting
        # ==========================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(strategy_df.index, strategy_df['Equity_LongOnly'], label='Strategy (Trend Filtered)', color='red', linewidth=2)
        ax1.plot(strategy_df.index, strategy_df['Equity_Bench'], label='Benchmark', color='gray', linestyle='--')
        ax1.set_title(f"Real-World Trading (Long Only)\nReturn: {total_ret*100:.2f}%", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(strategy_df.index, strategy_df['Equity_Alpha'], label='Pure Alpha (Hedged)', color='blue', linewidth=2)
        ax2.fill_between(strategy_df.index, strategy_df['Equity_Alpha'], 1, color='blue', alpha=0.1)
        ax2.set_title(f"Model Pure Alpha (Long vs Benchmark)\nExcess Return: {alpha_ret*100:.2f}%", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    backtester = BacktestSystem()
    backtester.run_backtest()