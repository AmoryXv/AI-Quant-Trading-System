import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr # 用来计算 RankIC

class QuantModeler:
    def __init__(self, data_path='./data_processed'):
        self.data_path = data_path
        
    def load_and_merge_data(self):
        """
        读取所有股票数据，合并成一个巨大的 DataFrame (Panel Data)
        """
        print("Loading data from parquet files...")
        file_list = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        
        all_data = []
        for f in file_list:
            df = pd.read_parquet(os.path.join(self.data_path, f))
            all_data.append(df)
            
        # 纵向合并：把300只股票的数据拼在一起
        # 现在的 shape 大约是 (300股票 * 250天, 特征数)
        dataset = pd.concat(all_data, ignore_index=True)
        
        # 简单清洗：去除无限值和空值
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 按时间排序 (为了后续按时间切分)
        dataset = dataset.sort_values('trade_date')
        
        print(f"Total Data Shape: {dataset.shape}")
        return dataset

    def train_xgboost(self):
        # 1. 准备数据
        df = self.load_and_merge_data()
        
        # 定义特征列 (X) 和 目标列 (Y)
        feature_cols = ['ROC_5', 'ROC_20', 'Vol_20', 'RSI', 'MACD_DIF', 'MACD_DEA', 'MACD_HIST']
        label_col = 'Future_Ret_5'
        
        # 2. 时间序列切分 (Train/Test Split)
        # 我们用 2023-11-01 之前的数据训练，之后的数据测试
        split_date = pd.to_datetime('2023-11-01')
        
        train_df = df[df['trade_date'] < split_date]
        test_df = df[df['trade_date'] >= split_date]
        
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]
        
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # 3. 建立并训练模型
        # XGBRegressor: 
        model = xgb.XGBRegressor(
            max_depth=4,            # 树深，防止过拟合
            learning_rate=0.05,     # 学习率
            n_estimators=100,       # 树的数量
            objective='reg:squarederror', 
            n_jobs=-1               # 并行计算
        )
        
        print("Start Training XGBoost...")
        model.fit(X_train, y_train)
        
        # 4. 预测与评估
        pred_test = model.predict(X_test)
        
        # 计算 IC (Information Coefficient) - 预测值与真实值的皮尔逊相关系数
        ic_score = np.corrcoef(y_test, pred_test)[0, 1]
        
        # 计算 RankIC - 预测排名与真实排名的斯皮尔曼相关系数 
        rank_ic_score, _ = spearmanr(y_test, pred_test)
        
        print("\n" + "="*30)
        print(f"Model Evaluation (Test Set):")
        print(f"IC Score:      {ic_score:.4f}")
        print(f"Rank IC Score: {rank_ic_score:.4f}")
        print("="*30 + "\n")
        
        # 5. 特征重要性分析 (Feature Importance)
        xgb.plot_importance(model)
        plt.title("Feature Importance")
        plt.show()
        
        return model

if __name__ == "__main__":
    modeler = QuantModeler()
    modeler.train_xgboost()