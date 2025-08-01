import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates

def portfolio_actions(reults_df):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(reults_df['Date'], reults_df['Close'], color='green', marker='o')
    
    state_colors = {
        'buy': 'lightgreen',
        'sell': 'lightcoral',
        'hold': 'lightyellow'
    }
    
    prev_state = None
    start_date = None
    
    for i, row in reults_df.iterrows():
        current_state = row['Color_Action']
        current_date = row['Date']
    
        if current_state != prev_state:
            if prev_state is not None:
                ax.axvspan(start_date, current_date, color=state_colors[prev_state], alpha=0.3)
            start_date = current_date
            prev_state = current_state
    
    ax.axvspan(start_date, reults_df['Date'].iloc[-1], color=state_colors[prev_state], alpha=0.3)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.title("Portfolio Value Behavior")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    
    skip = 100
    plt.xticks(reults_df['Date'][::skip], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def portfolio_value(reults_df):
    plt.figure(figsize=(10, 5))
    plt.plot(reults_df['Date'], reults_df['Portfolio_Values'], marker='o', linestyle='-', color='blue')
    plt.title('Portfolio Value Behavior')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    
    skip = 100
    plt.xticks(reults_df['Date'][::skip], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def portfolio_behavior(reults_df):
    plt.figure(figsize=(12, 6))

    plt.plot(reults_df['Date'], reults_df['Actual_Cash'], label='Cash', color='Yellow', marker='o')
    plt.plot(reults_df['Date'], reults_df['Stocks_Money'], label='Stocks Money', color='Green', marker='x')
    
    # Configuración general
    plt.title('Evolución de Cash y Acciones')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    
    skip = 100
    plt.xticks(reults_df['Date'][::skip], rotation=45)
    
    plt.tight_layout()
    
    # Mostrar
    plt.show()