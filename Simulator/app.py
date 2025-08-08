from datetime import datetime, timedelta
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dateutil.relativedelta import relativedelta
from Clases.extractData import fetch_stock_data
from Clases.generateFeatures import add_features
from Clases.tradingSimulation import trading_simulation
import pandas as pd
import numpy as np

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"Error in parameters format!! ")
        sys.exit(2)
        
def parse_date(date_str):
    """Convert YYYY-MM-DD to datetime.date"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: '{date_str}'. Use format YYYY-MM-DD.")
    
def animate_portfolio(results_df):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    dates = pd.to_datetime(results_df['Date'])
    cash = results_df['Actual_Cash']
    stocks = results_df['Stocks_Money']
    portfolio = results_df['Portfolio_Values']
    action = results_df['Prev_Action']
    benchmark = results_df['Benchmark']
    num_stocks = results_df['Stocks_Held']
    stock_price = results_df['Close']
    
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, max(portfolio.max(), stocks.max()) * 1.3)
    ax.grid(True)
    
    line_cash, = ax.plot([], [], 'b-o', label='Cash', markersize=4)
    line_stocks, = ax.plot([], [], 'g-s', label='Stocks', markersize=4)
    line_portfolio, = ax.plot([], [], 'r-', label='Portfolio Total', lw=2)
    
    info_text = ax.text(0.02, 0.87, '', transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    stock_text = ax.text(
        0.846, 1.02,  '', transform=ax.transAxes,
        bbox=dict(facecolor='whitesmoke', alpha=0.9), fontsize=9, ha='left')
    
    def init():
        line_cash.set_data([], [])
        line_stocks.set_data([], [])
        line_portfolio.set_data([], [])
        info_text.set_text('')
        stock_text.set_text('')
        return line_cash, line_stocks, line_portfolio, info_text, stock_text
    
    def update(frame):
        line_cash.set_data(dates[:frame+1], cash[:frame+1])
        line_stocks.set_data(dates[:frame+1], stocks[:frame+1])
        line_portfolio.set_data(dates[:frame+1], portfolio[:frame+1])
        
        current_date = dates.iloc[frame].strftime('%Y-%m-%d')
        current_value = portfolio.iloc[frame]
        current_benchmark = benchmark.iloc[frame]
        current_action = action.iloc[frame]
        current_cash = cash.iloc[frame]
        current_numStocks = num_stocks.iloc[frame]
        current_stock_price = stock_price.iloc[frame]
        
        info_text.set_text(
            f"Date: {current_date}\n"
            f"Action: {current_action}\n"
            f"Actual Cash: ${current_cash:,.2f}\n"
            f"Number Stocks: ${current_numStocks}\n"
            f"Portfolio: ${current_value:,.2f}\n"
            f"Benchmark: ${current_benchmark:,.2f}"
        )
        
        stock_text.set_text(
            f"Stock Price: {current_stock_price:,.2f}"
        )
        
        
        return line_cash, line_stocks, line_portfolio, info_text, stock_text
    
    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(results_df),
        init_func=init,
        interval=300,
        blit=True,
        repeat=False
    )
    
    plt.legend(loc='upper right')
    plt.title('Investment Behavior', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return ani

if __name__ == "__main__":
    parser = MyParser(description="Trading Simulation")
    parser.add_argument("--capital", type=float, required=True, help="USD initial capital")
    parser.add_argument("--start_date", type=parse_date, required=True, help="Start Date of Trading")
    parser.add_argument("--end_date", type=parse_date, required=True, help="End Date of Trading")
    parser.add_argument("--companie", type=str, required=True, help="Companie to invest")
    args = parser.parse_args()

    capital = args.capital
    start_date = args.start_date
    end_date = args.end_date
    companie = args.companie
    
    start_date_minus_1y = start_date - relativedelta(years=1)
    
    yf_df = fetch_stock_data(companie, start_date_minus_1y, end_date)
    inputs_df = add_features(yf_df)
    inputs_df = inputs_df[inputs_df['Date'] >= start_date]
    
    total_reward_up, final_value, reults_df = trading_simulation(inputs_df, capital)
    
    animate_portfolio(reults_df)
    