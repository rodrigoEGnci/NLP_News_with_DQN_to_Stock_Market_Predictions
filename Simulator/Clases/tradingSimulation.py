import numpy as np
import pandas as pd
import torch
import random
import math
from .tradingEnv import TradingEnv
from .DQN import DQN

def group_actions(valor):
    valor = valor.lower()
    if "buy" in valor:
        return "buy"
    elif "sell" in valor:
        return "sell"
    elif "hold" in valor:
        return "hold"
    
def trading_simulation(env_df, capital):
    test_env = TradingEnv(env_df, capital)
    trained_model = DQN(input_dim=17, output_dim=8)
    modelName = 'Models/dqn_trading_Experiment_3.pth'
    trained_model.load_state_dict(torch.load(modelName, map_location=torch.device('cpu')))
    trained_model.eval()

    state = test_env.reset()
    done = False
    total_reward = 0
    actions = []
    stocks_held = []
    actual_cash = []
    stock_price = []
    portfolio_values = []
    
    while not done:
    
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
        with torch.no_grad():
            action_values = trained_model(state_tensor)
    
        # print("Valores de acción (8 valores):", action_values)
        # print("Máxima acción posible:", torch.argmax(action_values).item())
        action = torch.argmax(action_values).item()
        #print(action)
        next_state, reward, done, _ = test_env.step(action)
    
        total_reward += reward
    
    
        actions.append(action)
        stocks_held.append(test_env.shares_held)
        actual_cash.append(test_env.cash)
        stock_price.append(test_env.current_price)
        portfolio_values.append(test_env.prev_portfolio_value)
    
        state = next_state
    reults_df = env_df[['Date','Close']].copy()
    reults_df = reults_df.drop(reults_df.index[0])
    reults_df = reults_df.drop(reults_df.tail(2).index)
    reults_df['v_Close'] = stock_price
    reults_df['Prev_Action'] = actions
    reults_df['Stocks_Held'] = stocks_held
    reults_df['Actual_Cash'] = actual_cash
    reults_df['Portfolio_Values'] = portfolio_values
    
    reults_df['Prev_Action'] = reults_df['Prev_Action'].map({
        0: "Buy Small (5%)",
        1: "Buy Small - Medium (10%)",
        2: "Buy Large - Medium (20%)",
        3: "Buy Large (35%)",
        4: "Sell Small (10%)",
        5: "Sell Small - Medium (25%)",
        6: "Sell Large - Medium (50%)",
        7: "Sell Large (All)"
    })
    
    reults_df['Stocks_Money'] = reults_df['Close'] * reults_df['Stocks_Held']
    reults_df['Color_Action'] = reults_df['Prev_Action'].apply(group_actions)
    
    initial_price = reults_df['Close'].iloc[0]
    reults_df['Benchmark'] = capital * ((reults_df['Close']) / initial_price)
    #normal_final_invest = 10000 * (test_vis_SPY['Close'].iloc[-1] / test_vis_SPY['Close'].iloc[0])
    
    return total_reward, test_env.prev_portfolio_value, reults_df