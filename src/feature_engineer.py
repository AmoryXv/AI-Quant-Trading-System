import pandas as pd
import numpy as np
import os
import warnings

# 忽略 pandas 的一些计算警告
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    特征工厂：负责计算技术指标（Factors）和预测目标（Labels）
    """
    
    def __init__(self, data_path='./data', output_path='./data_processed'):
        self.data_path = data_path
        self.output_path = output_path
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        计算 RSI (相对强弱指标) - 衡量买卖力量
        原理：N日内涨幅平均值 / (N日内涨幅平均值 + N日内跌幅平均值)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calc_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 MACD (异同移动平均线) - 衡量趋势
        """
        # EMA (指数移动平均)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['MACD_DIF'] = ema12 - ema26
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])
        return df

    def process_single_stock(self, filename: str):
        """
        处理单只股票的数据
        """
        # 1. 读取数据
        file_path = os.path.join(self.data_path, filename)
        df = pd.read_parquet(file_path)
        
        # 确保按时间排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # ================================
        # 2. 构建因子 (X)
        # ================================
        
        # [Factor 1] ROC (Rate of Change): 过去N天的收益率
        # 动量因子：过去强的大概率未来继续强（或反转）
        df['ROC_5'] = df['close'].pct_change(5)
        df['ROC_20'] = df['close'].pct_change(20)
        
        # [Factor 2] Volatility (波动率): 过去20天的标准差
        # 风险因子：波动越大，风险越大
        df['Vol_20'] = df['close'].rolling(20).std()
        
        # [Factor 3] RSI
        df['RSI'] = self._calc_rsi(df['close'])
        
        # [Factor 4] MACD
        df = self._calc_macd(df)
        
        # ================================
        # 3. 构建标签 (Y) -> 核心！
        # ================================
        # 目标：预测“未来 5 天”的收益率
        # shift(-5) 把未来的数据向上平移，让我们在“今天”这一行能看到“未来”的结果
        df['Future_Ret_5'] = df['close'].shift(-5) / df['close'] - 1.0
        
        # ================================
        # 4. 清洗与落盘
        # ================================
        # 去除因为 rolling 计算产生的 NaN 值（前几十行通常是空的）
        # 注意：最后 5 行的 Label 也会是 NaN（因为没有未来了），也要去掉
        df = df.dropna()
        
        if not df.empty:
            save_path = os.path.join(self.output_path, filename)
            df.to_parquet(save_path)
            # print(f"Processed {filename}") # 太多打印会刷屏，先注释掉

    def run_batch(self):
        """批量处理所有文件"""
        file_list = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        total = len(file_list)
        print(f"Start feature engineering for {total} stocks...")
        
        for i, f in enumerate(file_list):
            self.process_single_stock(f)
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1}/{total}")
                
        print("Feature engineering finished!")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_batch()
    
    # 简单的验证：读取一个处理后的文件看看样子
    test_file = os.listdir('./data_processed')[0]
    df_test = pd.read_parquet(os.path.join('./data_processed', test_file))
    
    print("\n[Preview Data Structure]")
    print(df_test[['trade_date', 'close', 'RSI', 'Vol_20', 'Future_Ret_5']].head())