# src/backtest_engine.py
# Final Optimized Version: Powered by Optuna & Walk-Forward Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import warnings
import joblib
import torch
from dateutil.relativedelta import relativedelta
from xgboost import XGBRegressor
from model_lstm import LSTMPredictor

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class BacktestSystem:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_root, 'data_processed')
        self.dll_path = os.path.join(self.project_root, 'strategy.dll')
        
        if not os.path.exists(self.dll_path):
             raise FileNotFoundError(f"Critical: {self.dll_path} missing!")
        
        try:
            self.lib = ctypes.CDLL(self.dll_path)
            self.lib.generate_signals.argtypes = [
                ctypes.POINTER(ctypes.c_double), 
                ctypes.POINTER(ctypes.c_double), 
                ctypes.c_int,                    
                ctypes.c_double                  
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
        full_df = self.load_data()
        feature_cols = ['ROC_5', 'ROC_20', 'Vol_20', 'RSI', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST']
        target_col = 'Future_Ret_5'

        # ==========================================
        # 1. AI Optimized Parameters (from Optuna)
        # ==========================================
        BEST_PARAMS = {
            'hidden_dim': 64,
            'learning_rate': 0.00014854,
            'epochs': 18,
            'ma_window': 15,            
            'sentiment_threshold': 0.001173, 
            'stock_threshold': 0.00396,      # 个股涨幅门槛
            'top_k': 3
        }
        
        # ==========================================
        # 2. Smart Date Logic (Walk-Forward Setup)
        # ==========================================
        min_date = full_df['trade_date'].min()
        max_date = full_df['trade_date'].max()
        
        # 训练窗口长度：6个月 (适应短数据)
        train_window_months = 6 
        test_window_months = 1
        
        earliest_possible_start = min_date + relativedelta(months=train_window_months)
        preferred_start = pd.to_datetime('2023-01-01')
        
        if earliest_possible_start > max_date:
            raise ValueError(f"Dataset is too short! Need at least {train_window_months} months.")
        
        if preferred_start < earliest_possible_start:
            start_date = earliest_possible_start
        else:
            start_date = preferred_start
            
        print(f"[System] Walk-Forward Start Date: {start_date.date()}")
        print(f"  - Params: {BEST_PARAMS}")

        prediction_fragments = []
        current_test_start = start_date

        # ==========================================
        # 3. Rolling Loop (时间循环)
        # ==========================================
        loop_counter = 1
        
        while current_test_start < max_date:
            current_test_end = current_test_start + relativedelta(months=test_window_months)
            current_train_start = current_test_start - relativedelta(months=train_window_months)
            
            train_mask = (full_df['trade_date'] >= current_train_start) & (full_df['trade_date'] < current_test_start)
            test_mask = (full_df['trade_date'] >= current_test_start) & (full_df['trade_date'] < current_test_end)
            
            window_train = full_df[train_mask].copy()
            window_test = full_df[test_mask].copy()
            
            if len(window_test) == 0:
                current_test_start = current_test_end
                continue
            if len(window_train) < 100:
                current_test_start = current_test_end
                continue

            print(f"\n>>> Window {loop_counter}: Test [{current_test_start.date()} ~ {current_test_end.date()}]")

            # --- Model Retraining using BEST PARAMS ---
            lstm_model = LSTMPredictor(
                sequence_length=10, 
                epochs=BEST_PARAMS['epochs'],           
                hidden_dim=BEST_PARAMS['hidden_dim'],   
                batch_size=1024,
                learning_rate=BEST_PARAMS['learning_rate'] 
            )
            
            try:
                lstm_model.fit(window_train, feature_cols, target_col)
                preds = lstm_model.predict(window_test, feature_cols)
                window_test['pred_lstm'] = preds
                prediction_fragments.append(window_test)
            except Exception as e:
                print(f"    [Error] Training failed: {e}")
            
            current_test_start = current_test_end
            loop_counter += 1

        # ==========================================
        # 4. Concatenate & Strategy Logic
        # ==========================================
        print("\n[System] Rolling Analysis Complete. Applying Optimized Strategy...")
        
        if not prediction_fragments:
            print("[Error] No predictions generated.")
            return
            
        test_df = pd.concat(prediction_fragments).sort_values('trade_date')
        
        # IC Evaluation
        eval_df = test_df.dropna(subset=['pred_lstm', target_col])
        if len(eval_df) > 0:
            ic_lstm = eval_df[['pred_lstm', target_col]].corr(method='spearman').iloc[0,1]
            print("\n" + "="*45)
            print(f" Optimized Rank IC: {ic_lstm:.4f}")
            print("="*45 + "\n")

        # --- Strategy Application (Using Optimized Parameters) ---
        
        # 1. 计算 MA (使用优化后的窗口 15)
        market_index = test_df.groupby('trade_date')['close'].mean()
        market_ma = market_index.rolling(window=BEST_PARAMS['ma_window']).mean() # MA 15
        test_df['market_price'] = test_df['trade_date'].map(market_index)
        test_df['market_ma'] = test_df['trade_date'].map(market_ma)
        
        # 2. C++ Filter
        final_preds = test_df['pred_lstm'].fillna(-999).values
        c_signals = np.zeros(len(final_preds), dtype=np.float64)
        self.lib.generate_signals(
            final_preds.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            c_signals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            len(final_preds), 
            ctypes.c_double(0.0) 
        )
        test_df['Signal_Raw'] = c_signals 

        # 3. Python Filter (Using Optimized Thresholds)
        test_df['Rank'] = test_df.groupby('trade_date')['pred_lstm'].rank(ascending=False, method='first')
        market_sentiment = test_df.groupby('trade_date')['pred_lstm'].transform('mean')
        trend_condition = test_df['market_price'] > test_df['market_ma']
        
        condition = (
            (test_df['Signal_Raw'] == 1) & 
            (test_df['Rank'] <= BEST_PARAMS['top_k']) &              # Top 3
            (test_df['pred_lstm'] > BEST_PARAMS['stock_threshold']) & # 个股 > 0.00396
            (market_sentiment > BEST_PARAMS['sentiment_threshold']) & # 大盘 > 0.00117
            (trend_condition)                                         # 站上 MA15
        )
        
        test_df['Final_Signal'] = np.where(condition, 1.0, 0.0)
        
        trade_days = test_df[test_df['Final_Signal'] == 1.0]['trade_date'].nunique()
        print(f">> Strategy Active Days: {trade_days} / {test_df['trade_date'].nunique()}")

        # ==========================================
        # 5. Performance & Plotting
        # ==========================================
        all_dates = np.sort(test_df['trade_date'].unique())
        daily_gross_ret = test_df[test_df['Final_Signal'] == 1.0].groupby('trade_date')['Future_Ret_5'].mean()
        benchmark_ret = test_df.groupby('trade_date')['Future_Ret_5'].mean()
        
        strategy_df = pd.DataFrame(index=all_dates)
        strategy_df['Benchmark'] = benchmark_ret
        strategy_df['Gross_Strategy'] = daily_gross_ret
        
        strategy_df['Gross_Strategy'] = strategy_df['Gross_Strategy'].fillna(0.0)
        strategy_df['Benchmark'] = strategy_df['Benchmark'].fillna(0.0)

        FRICTION_COST = 0.002 
        strategy_df['Net_Strategy'] = np.where(
            strategy_df['Gross_Strategy'] != 0, 
            (strategy_df['Gross_Strategy'] - FRICTION_COST), 
            0.0
        )

        strategy_df['Equity_LongOnly'] = (1 + strategy_df['Net_Strategy']/5).cumprod()
        strategy_df['Equity_Bench'] = (1 + strategy_df['Benchmark']/5).cumprod()
        strategy_df['Alpha_Ret'] = (strategy_df['Net_Strategy'] - strategy_df['Benchmark']) / 5
        strategy_df['Equity_Alpha'] = (1 + strategy_df['Alpha_Ret']).cumprod()

        total_ret = strategy_df['Equity_LongOnly'].iloc[-1] - 1
        alpha_ret = strategy_df['Equity_Alpha'].iloc[-1] - 1
        
        print("\n" + "="*45)
        print(f" [Final Optimized Performance]")
        print(f" Long-Only Return: {total_ret*100:6.2f}%")
        print(f" Hedged Alpha:     {alpha_ret*100:6.2f}%")
        print("="*45 + "\n")
        
        # Save Last Model
        model_dir = os.path.join(self.project_root, 'models')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(lstm_model.model.state_dict(), os.path.join(model_dir, 'lstm_optimized.pth'))
        joblib.dump(lstm_model.scaler_X, os.path.join(model_dir, 'scaler_x.pkl'))
        print(f"[System] Optimized model saved to {model_dir}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(strategy_df.index, strategy_df['Equity_LongOnly'], label='Net Strategy (Optimized)', color='red', linewidth=2)
        ax1.plot(strategy_df.index, strategy_df['Equity_Bench'], label='Benchmark', color='gray', linestyle='--')
        ax1.set_title(f"Real-World Trading (Optimized)\nReturn: {total_ret*100:.2f}%", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(strategy_df.index, strategy_df['Equity_Alpha'], label='Pure Alpha (Hedged)', color='blue', linewidth=2)
        ax2.fill_between(strategy_df.index, strategy_df['Equity_Alpha'], 1, color='blue', alpha=0.1)
        ax2.set_title(f"Model Pure Alpha (Optimized)\nExcess Return: {alpha_ret*100:.2f}%", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    backtester = BacktestSystem()
    backtester.run_backtest()