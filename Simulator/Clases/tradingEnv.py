import numpy as np
import pandas as pd
from collections import deque
import math

#Custom Trading Environment for a single stock
class TradingEnv:
    """
    Initialize the trading environment.
    """
    def __init__(self, df, initial_cash=10_000):
        # Load and sort data
        self.df = df.reset_index(drop=True).copy()
        self.max_steps = len(self.df) - 2  # total steps (days)

        # Initial portfolio settings
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares_held = 0.0
        self.current_step = 0
        self.prev_portfolio_value = initial_cash
        self.returns_history = []

        # Track total reward and history
        self.total_reward = 0
        self.history = []

        # Action mapping
        self.action_mapping = {
            0: 0.05,  # buy small
            1: 0.10,  # buy small - medium
            2: 0.20,  # buy large - medium
            3: 0.35,  # buy large
            4: 0.10,  # sell small
            5: 0.25,  # sell small - medium
            6: 0.50,  # sell large - medium
            7: 1.00   # sell all
        }

        # Define input features
        self.feature_cols = [
            'Close',
            'SMA_5', 'SMA_20', 'SMA_50',
            'RSI_14',
            'MACD', 'MACD_Signal',
            'Bollinger_b',
            'ATR_14',
            'Momentum_10',
            'Relative_price', 'Norm_Volatily', 'Scal_RSI'
        ]

    """
    Reset the environment to the initial state.
    """
    def reset(self):
        self.cash = self.initial_cash          # Reset available capital
        self.shares_held = 0.0                 # No shares at start
        self.current_step = 0                  # Start at the beginning of the dataset
        self.prev_portfolio_value = self.initial_cash  # Track portfolio for reward calc
        self.total_reward = 0.0                # Reset reward tracker
        self.history = []
        self.returns_history = []# Clear history

        return self._get_state()

    """
    Construct the current state vector.
    """
    def _get_state(self):
        row = self.df.loc[self.current_step]

        # Extract the market features
        features = []
        for col in self.feature_cols:
            features.append(row[col])

        # Append portfolio state
        current_price = self.df.loc[self.current_step, 'Close']
        features.append(self.shares_held * current_price / self.cash)  # ratio position/cash (ADD)
        features.append(self.shares_held * current_price / self.initial_cash)  # % invest portfolio (ADD)
        features.append(self.cash)
        features.append(self.shares_held)

        return np.array(features, dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        invalid_action_penalty = -.01

        current_price = self.df.loc[self.current_step, 'Close']
        action_type = self.action_mapping[action]

        #Make Step
        if 0 <= action <= 3:  # buy
            percent = self.action_mapping[action]
            if self.cash >= current_price * action_type:
                self._buy(percent)
        elif 4 <= action <= 7:  # sell
            percent = self.action_mapping[action]
            if self.shares_held > 0:
                self._sell(percent)

        #Get portfolio Value
        portfolio_value = self._get_portfolio_value()

        #Calculate Daily Return
        daily_return = (portfolio_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-6)
        self.returns_history.append(daily_return)

        if len(self.returns_history) > 100:
            self.returns_history.pop(0)

        #Update peak_portfolio
        #self.peak_portfolio = max(self.peak_portfolio, portfolio_value)

        # Calculate reward
        reward += self.calculate_reward(action)
        self.total_reward += reward

        # Advance to next time step
        self.current_step += 1

        #Get new Price of stock
        self.current_price = self.df.loc[self.current_step, 'Close']

        # Update prev_portfolio value
        self.prev_portfolio_value = portfolio_value

        # Log history for debugging/analysis
        self.history.append({
            'step': self.current_step,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'portfolio_value': portfolio_value,
            'action': action,
            'reward': reward
        })

        done = self.current_step >= self.max_steps - 1
        if done:

            next_state = self._get_state()
            return next_state, reward, done, {}


        next_state = self._get_state()
        return next_state, reward, done, {}


    def calculate_reward(self, action):
        # Inicializar componentes
        rewards = {
            'pnl': 0.0,                  # Profit & Loss básico
            'risk_adjusted': 0.0,        # Retorno ajustado por riesgo
            'invalid_action': 0.0,       # Penalización por acción inválida
            'position_management': 0.0,  # Manejo de posición óptimo
            'trend_alignment': 0.0      # Alineación con tendencia
        }

        # 1. Componente PnL (base)
        current_value = self._get_portfolio_value()
        pnl = (current_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-6)
        rewards['pnl'] = np.clip(pnl * 5, -2.0, 2.0)  # Escalado y limitado

        # 2. Componente Ajustado por Riesgo (Sharpe-like)
        returns_window = np.array(self.returns_history[-20:])  # Últimos 20 retornos
        if len(returns_window) > 5:
            volatility = np.std(returns_window) + 1e-6
            risk_free = 0.0002  # Tasa libre de riesgo diaria aprox.
            sharpe_like = (np.mean(returns_window) - risk_free) / volatility
            rewards['risk_adjusted'] = np.clip(sharpe_like * 2, -1.5, 1.5)

        # 3. Penalización por Acción Inválida (¡Nueva recomendación!)
        invalid_penalty = self._get_invalid_action_penalty(action)
        rewards['invalid_action'] = invalid_penalty

        # 4. Gestión de Posición
        rewards['position_management'] = self._calculate_position_score()

        # 5. Alineación con Tendencia
        rewards['trend_alignment'] = self._calculate_trend_alignment()

        # Ponderación final
        weights = {
            'pnl': 0.4,
            'risk_adjusted': 0.3,
            'invalid_action': 0.15,
            'position_management': 0.1,
            'trend_alignment': 0.05
        }

        total_reward = sum(rewards[component] * weights[component] for component in rewards)
        return np.clip(total_reward, -3.0, 3.0)  # Limitar para estabilidad


    def _get_invalid_action_penalty(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        action_type = self.action_mapping[action]

        # Penalización base escalonada
        if (0 <= action <= 3) and (self.cash < current_price * action_type):
            # Intento de compra sin fondos
            base_penalty = -0.5

            # Penalización adicional proporcional a lo "ambicioso" de la acción
            ambition_penalty = -0.3 * action_type  # Más fuerte para acciones mayores

            return base_penalty + ambition_penalty

        elif (4 <= action <= 7) and (self.shares_held <= 0):
            # Intento de venta sin posición
            base_penalty = -0.4

            # Penalización por tamaño de orden de venta
            size_penalty = -0.2 * abs(action_type)

            return base_penalty + size_penalty

        return 0.0  # Acción válida

    def _calculate_position_score(self):
        """Evalúa qué tan óptima es la posición actual"""
        current_price = self.df.loc[self.current_step, 'Close']
        position_ratio = (self.shares_held * current_price) / self._get_portfolio_value()

        # Ideal: 30-70% invertido (evita estar totalmente en cash o totalmente invertido)
        optimal_min = 0.3
        optimal_max = 0.7

        if position_ratio < optimal_min:
            return -0.5 * (optimal_min - position_ratio)  # Penaliza estar bajo-invertido
        elif position_ratio > optimal_max:
            return -0.8 * (position_ratio - optimal_max)  # Penaliza más el sobre-invertir
        else:
            return 0.3  # Recompensa por estar en rango óptimo

    def _calculate_trend_alignment(self):
        """Recompensa alinear acciones con tendencia del mercado"""
        price_vs_sma = self.df.loc[self.current_step, 'Close'] / self.df.loc[self.current_step, 'SMA_20']
        rsi = self.df.loc[self.current_step, 'RSI_14']

        # Lógica de tendencia
        if price_vs_sma > 1.02 and rsi < 70:  # Tendencia alcista saludable
            return 0.4 if self.shares_held > 0 else -0.3
        elif price_vs_sma < 0.98 and rsi > 30:  # Tendencia bajista
            return 0.3 if self.shares_held == 0 else -0.4
        else:
            return 0.1 if abs(self.shares_held) < 0.1 else 0.0  # Mercado lateral

    def _get_portfolio_value(self):
        current_price = self.df.loc[self.current_step, 'Close']
        return self.cash + (self.shares_held * current_price)

    """
    Execute a buy order using a percentage of available cash.
    """
    def _buy(self, percent):
        current_price = self.df.loc[self.current_step, 'Close']

        # Capital to use for this transaction
        amount_to_spend = self.cash * percent

        # Number of whole shares we can buy
        shares_to_buy = int(amount_to_spend // current_price)

        if shares_to_buy > 0:
            self.cash -= shares_to_buy * current_price
            self.shares_held += shares_to_buy

    """
    Execute a sell order using a percentage of held shares.
    """
    def _sell(self, percent):
        current_price = self.df.loc[self.current_step, 'Close']

        # Determine how many shares to sell
        shares_to_sell = int(self.shares_held * percent)

        if shares_to_sell > 0:
            self.cash += shares_to_sell * current_price
            self.shares_held -= shares_to_sell
