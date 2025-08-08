import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df['Ticker'] = ticker 
        return df
    except Exception as e:
        print(f"Error in connection with Yahoo Finance...")
        return None