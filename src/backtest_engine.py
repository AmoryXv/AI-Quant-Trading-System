import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import sys
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class BacktestSystem:
    def __init__(self):
        # ==========================================
        # 路径修复核心逻辑 (Path Fixing Logic)
        # ==========================================
        # 1. 获取当前脚本所在目录 (src)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 获取项目根目录 (src 的上一级)
        self.project_root = os.path.dirname(current_dir)
        
        # 3. 拼接数据路径 (Root + data_processed)
        self.data_path = os.path.join(self.project_root, 'data_processed')
        
        # 4. 拼接 DLL 路径 (Root + strategy.dll)
        self.dll_path = os.path.join(self.project_root, 'strategy.dll')
        
        print(f"[Info] Project Root: {self.project_root}")
        print(f"[Info] DLL Path: {self.dll_path}")

        # 检查 DLL 是否存在
        if not os.path.exists(self.dll_path):
             # 如果找不到，提示用户可能需要重新编译
            raise FileNotFoundError(f"Critical Error: strategy.dll not found at {self.dll_path}. \nPlease run 'python src/build_cpp.py' from root!")

        try:
            self.lib = ctypes.CDLL(self.dll_path)
            print(f"[Engine] Success: C++ Strategy Module Loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to link C++ Library: {e}")

    def load_data(self):
        print("Loading data...")
        file_list = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        all_data = []
        for f in file_list:
            df = pd.read_parquet(os.path.join(self.data_path, f))
            all_data.append(df)
        
        dataset = pd.concat(all_data, ignore_index=True)
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        dataset = dataset.sort_values('trade_date')
        return dataset

    def run_backtest(self):
        df = self.load_data()
        
        # 1. 切分数据
        split_date = pd.to_datetime('2023-10-01')
        train_df = df[df['trade_date'] < split_date]
        test_df = df[df['trade_date'] >= split_date].copy()
        
        feature_cols = ['ROC_5', 'ROC_20', 'Vol_20', 'RSI', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST']
        
        # 2. 训练模型
        print(f"Training XGBoost on {len(train_df)} samples...")
        self.model = XGBRegressor(max_depth=4, n_estimators=50, n_jobs=-1, random_state=42)
        self.model.fit(train_df[feature_cols], train_df['Future_Ret_5'])
        
        # 3. 预测
        print("Generating Predictions...")
        predictions = self.model.predict(test_df[feature_cols])
        
        # ==========================================
        # 4. 策略生成 (纯 C++ 通道)
        # ==========================================
        THRESHOLD = 0.01
        
        # 准备 C++ 接口数据
        data_len = len(predictions)
        self.lib.generate_signals.argtypes = [
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.c_int,                    
            ctypes.c_double                  
        ]
        
        # Numpy -> C Pointer 高效转换
        c_preds = predictions.astype(np.float64)
        c_signals = np.zeros(data_len, dtype=np.float64)
        
        preds_ptr = c_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        signals_ptr = c_signals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # 调用 DLL
        self.lib.generate_signals(preds_ptr, signals_ptr, data_len, THRESHOLD)
        
        # 存回 DataFrame (Top-K 策略前置过滤)
        test_df['Signal_Raw'] = c_signals 
        print(">> Signal Generation: C++ Accelerated")

        # ==========================================
        # 5. 结合 Python 做 Top 5 组合逻辑
        # ==========================================
        # C++ 负责“初筛”(过滤掉预测值低的)，Python 负责“排序和组合”(复杂的业务逻辑)
        # 这种 "C++做计算，Python做业务" 是最架构师思维的写法
        
        test_df['Prediction'] = predictions
        # 只有 C++ 说能买(Signal_Raw=1)的股票，我们才拿来排名
        test_df['Rank'] = test_df.groupby('trade_date')['Prediction'].rank(ascending=False, method='first')
        
        # 最终持仓信号：C++初筛通过 且 排名在前5
        test_df['Final_Signal'] = np.where((test_df['Signal_Raw'] == 1) & (test_df['Rank'] <= 5), 1.0, 0.0)
        
        # 计算资金曲线
        daily_perf = test_df[test_df['Final_Signal'] == 1.0].groupby('trade_date')['Future_Ret_5'].mean()
        benchmark_perf = test_df.groupby('trade_date')['Future_Ret_5'].mean()
        
        strategy_df = pd.DataFrame({'Strategy_Ret': daily_perf, 'Benchmark_Ret': benchmark_perf}).fillna(0)
        strategy_df['Algo_Equity'] = (1 + strategy_df['Strategy_Ret'] / 5).cumprod()
        strategy_df['Bench_Equity'] = (1 + strategy_df['Benchmark_Ret'] / 5).cumprod()
        
        total_ret = strategy_df['Algo_Equity'].iloc[-1] - 1
        sharpe = (strategy_df['Strategy_Ret'].mean() / strategy_df['Strategy_Ret'].std()) * np.sqrt(250)
        
        print("\n" + "="*40)
        print(f" [Final C++ Hybrid Result]")
        print(f" Total Return:   {total_ret*100:6.2f}%")
        print(f" Sharpe Ratio:   {sharpe:6.2f}")
        print("="*40 + "\n")
        
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_df.index, strategy_df['Algo_Equity'], label='C++ Hybrid Strategy', color='#d62728')
        plt.plot(strategy_df.index, strategy_df['Bench_Equity'], label='Benchmark', color='gray', linestyle='--')
        plt.title(f"Hybrid Quant Strategy (C++ Core)\nSharpe: {sharpe:.2f}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    backtester = BacktestSystem()
    backtester.run_backtest()