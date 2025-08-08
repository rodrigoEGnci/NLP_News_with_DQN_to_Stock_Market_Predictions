import pandas as pd
import numpy as np
import math
from dateutil.relativedelta import relativedelta

def add_features(SPY_stocks):
    inputs_SPY = SPY_stocks.copy()
    if 'Date' not in inputs_SPY.columns and isinstance(inputs_SPY.index, pd.DatetimeIndex):
        inputs_SPY = inputs_SPY.reset_index()
        
    inputs_SPY['Date'] = inputs_SPY['Date'].dt.date
    inputs_SPY = inputs_SPY[['Date', 'Close', 'Low', 'High']]

    # --- Simple Moving Averages ---
    inputs_SPY['SMA_5'] = inputs_SPY['Close'].rolling(window=5).mean()
    inputs_SPY['SMA_20'] = inputs_SPY['Close'].rolling(window=20).mean()
    inputs_SPY['SMA_50'] = inputs_SPY['Close'].rolling(window=50).mean()
    
    # --- RSI (14) ---
    delta = inputs_SPY['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    inputs_SPY['RSI_14'] = 100 - (100 / (1 + rs))
    
    # --- MACD and Signal ---
    ema_12 = inputs_SPY['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = inputs_SPY['Close'].ewm(span=26, adjust=False).mean()
    inputs_SPY['MACD'] = ema_12 - ema_26
    inputs_SPY['MACD_Signal'] = inputs_SPY['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- Bollinger Bands (%B) ---
    sma_20 = inputs_SPY['Close'].rolling(window=20).mean()
    std_20 = inputs_SPY['Close'].rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    inputs_SPY['Bollinger_b'] = (inputs_SPY['Close'] - lower_band) / (upper_band - lower_band + 1e-10)
    
    # --- ATR (14) ---
    high_low = inputs_SPY['High'] - inputs_SPY['Low']
    high_close = np.abs(inputs_SPY['High'] - inputs_SPY['Close'].shift())
    low_close = np.abs(inputs_SPY['Low'] - inputs_SPY['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    inputs_SPY['ATR_14'] = true_range.rolling(window=14).mean()
    
    # --- Momentum (10) ---
    inputs_SPY['Momentum_10'] = inputs_SPY['Close'] - inputs_SPY['Close'].shift(10)

    # --- Add Experiment 3 features ---
    inputs_SPY['Relative_price'] = inputs_SPY['Close']/inputs_SPY['SMA_20']
    inputs_SPY['Norm_Volatily'] = inputs_SPY['ATR_14']/inputs_SPY['Close']
    inputs_SPY['Scal_RSI'] = inputs_SPY['RSI_14']/100

    inputs_SPY = inputs_SPY.drop(['Low', 'High'], axis=1)
    
    return inputs_SPY