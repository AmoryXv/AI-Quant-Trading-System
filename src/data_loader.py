import baostock as bs
import pandas as pd
import os
import time

class MarketDataLoader:
    def __init__(self, data_save_path: str = './data'):
        self.data_path = data_save_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        # 登陆系统
        bs.login()

    def fetch_hs300_stocks(self, date: str = '2023-12-31') -> list:
        """
        获取沪深300成分股名单
        :param date: 查询日期
        :return: 股票代码列表 ['sh.600000', 'sz.000001', ...]
        """
        print(f"Getting HS300 stocks list for date: {date}...")
        rs = bs.query_hs300_stocks(date=date)
        
        hs300_stocks = []
        while rs.next():
            hs300_stocks.append(rs.get_row_data()[1]) # index 1 is code
            
        print(f"Found {len(hs300_stocks)} stocks in HS300.")
        return hs300_stocks

    def fetch_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 指定获取的字段
        fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
        
        rs = bs.query_history_k_data_plus(stock_code, fields,
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag='2') 
        
        if rs.error_code != '0':
            return pd.DataFrame()

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据清洗
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'code': 'ts_code', 'date': 'trade_date'})
        return df

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        full_path = os.path.join(self.data_path, filename)
        df.to_parquet(full_path, engine='pyarrow')

# ==========================================
# 主程序：批量下载引擎
# ==========================================
if __name__ == "__main__":
    loader = MarketDataLoader()
    
    # 1. 获取股票列表
    stock_pool = loader.fetch_hs300_stocks()
    
    # 为了演示，我们先只下载前 5 只股票测试 (以免你等太久)
    # 确认没问题后，你可以把 [:5] 去掉，跑全量
    mini_pool = stock_pool
    
    print(f"Start downloading {len(mini_pool)} stocks...")
    
    for i, stock in enumerate(mini_pool):
        print(f"[{i+1}/{len(mini_pool)}] Processing {stock}...")
        
        df = loader.fetch_daily_data(stock, 
                                     start_date='2023-01-01', 
                                     end_date='2023-12-31')
        
        if not df.empty:
            # 文件名处理: sz.000001 -> sz000001
            safe_name = stock.replace('.', '') 
            loader.save_to_parquet(df, f'{safe_name}_2023.parquet')
            
    # 程序结束时显式退出，不再依赖 __del__，消除报错
    bs.logout()
    print("All tasks finished.")