import numpy as np
import pandas as pd

def evaluations(df):
    # Assuming your DataFrame is named 'df'
    # Calculate daily returns first
    df['daily_returns'] = df['Portfolio_Values'].pct_change()
    
    # 1. Cumulative Return
    initial_portfolio = df['Portfolio_Values'].iloc[0]
    final_portfolio = df['Portfolio_Values'].iloc[-1]
    cumulative_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
    
    # 2. Maximum Drawdown (MDD)
    running_max = df['Portfolio_Values'].cummax()
    daily_drawdown = (running_max - df['Portfolio_Values']) / running_max
    max_drawdown = daily_drawdown.max() * 100
    
    # 3. Annualized Sharpe Ratio
    risk_free_rate = 0.0  # Adjust if needed
    sharpe_ratio = np.sqrt(252) * (df['daily_returns'].mean() - risk_free_rate) / df['daily_returns'].std()
    
    # 4. Profit Factor 
    df['trade_result'] = df['Portfolio_Values'].diff()  # Change in portfolio value
    gross_profit = df[df['trade_result'] > 0]['trade_result'].sum()
    gross_loss = abs(df[df['trade_result'] < 0]['trade_result'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    # 5. Win Rate
    win_rate = (df['trade_result']).mean() 

    results = [initial_portfolio, final_portfolio, cumulative_return, max_drawdown, sharpe_ratio, profit_factor, win_rate]
    rounded_list = [round(val, 2) if isinstance(val, (int, float)) else val for val in results]
    
    return rounded_list

def benchmark(df):
    
    normal_final_invest = 10000 * (df['Close'].iloc[-1] / df['Close'].iloc[0])
    
    # Assuming your DataFrame is named 'df' with 'Date' and 'Close' columns
    # Calculate daily returns
    df['daily_returns'] = df['Close'].pct_change()
    
    # 1. Cumulative Return
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    cumulative_return = (final_price - initial_price) / initial_price * 100
    
    # 2. Maximum Drawdown (MDD)
    running_max = df['Close'].cummax()
    daily_drawdown = (running_max - df['Close']) / running_max
    max_drawdown = daily_drawdown.max() * 100
    
    # 3. Annualized Sharpe Ratio
    risk_free_rate = 0.0  # Adjust if needed
    sharpe_ratio = np.sqrt(252) * (df['daily_returns'].mean() - risk_free_rate) / df['daily_returns'].std()
    
    # 4. Profit Factor (not applicable for buy-and-hold) Requires trade data
    # 5. Win Rate (not applicable for buy-and-hold) Requires trade data

    results = [10000, normal_final_invest, cumulative_return, max_drawdown, sharpe_ratio]
    rounded_list = [round(val, 2) if isinstance(val, (int, float)) else val for val in results]
    
    return rounded_list