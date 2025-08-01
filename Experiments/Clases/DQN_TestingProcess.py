import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import math
from .DQN_TradingEnvNews import TradingEnv
from .DQN_TradingEnvE3 import TradingEnvE3
from .DQN import DQN
from .DQN_Agent import Agent
from .DQN_ReplayMemory import ReplayMemory

def group_actions(valor):
    valor = valor.lower()
    if "buy" in valor:
        return "buy"
    elif "sell" in valor:
        return "sell"
    elif "hold" in valor:
        return "hold"
    
def trading_simulation(env_df, env_name, inNN, outNN, modelName):
    if 'e3' in env_name:
        test_env = TradingEnvE3(env_df)
    elif 'news' in env_name:
        test_env = TradingEnv(env_df)
        
    trained_model = DQN(input_dim=inNN, output_dim=outNN)
    #trained_model.load_state_dict(torch.load('Models/dqn_MaxReward_trading_Experiment_6.pth'))
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
        0: "Buy Small",
        1: "Buy Small - Medium",
        2: "Buy Large - Medium",
        3: "Buy Large",
        4: "Sell Small",
        5: "Sell Small - Medium",
        6: "Sell Large - Medium",
        7: "Sell Large"
    })
    
    reults_df['Stocks_Money'] = reults_df['Close'] * reults_df['Stocks_Held']
    reults_df['Color_Action'] = reults_df['Prev_Action'].apply(group_actions)
    
    return total_reward, test_env.prev_portfolio_value, reults_df