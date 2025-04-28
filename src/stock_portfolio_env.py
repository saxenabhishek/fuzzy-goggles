import gymnasium as gym
from gymnasium import spaces
import numpy as np
from io import StringIO


class StockPortfolioEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, stock_data, window=5, initial_cash=1000):
        super(StockPortfolioEnv, self).__init__()

        self.stock_data = stock_data
        self.stock_symbols = list(stock_data.keys())
        self.n_stocks = len(self.stock_symbols)
        self.window = window
        self.cash = initial_cash

        self.n_features_per_stock = 3 * window + 1  # Closing prices + holding

        self.current_step = {symbol: window - 1 for symbol in self.stock_symbols}
        self.holdings = {
            symbol: 0 for symbol in self.stock_symbols
        }  # 0: not holding, 1: holding
        self.purchase_price = {
            symbol: 0.0 for symbol in self.stock_symbols
        }  # To track buying price

        # Define action space: [-1, 0, 1] for each stock (sell, hold, buy)
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Define observation space: past window closing prices and holdings for all stocks
        low = np.full(self.n_stocks * self.n_features_per_stock, 0, dtype=np.float32)
        high = np.full(
            self.n_stocks * self.n_features_per_stock, 1000, dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Ensure DataFrames are sorted by date
        for symbol in self.stock_symbols:
            self.stock_data[symbol] = (
                self.stock_data[symbol].sort_values(by="date").reset_index(drop=True)
            )
            if len(self.stock_data[symbol]) < self.window:
                raise ValueError(
                    f"DataFrame for {symbol} length must be at least {self.window}"
                )

        self.max_steps = (
            min(len(self.stock_data[symbol]) for symbol in self.stock_symbols) - 1
        )

        self.buy_signals = {symbol: [] for symbol in self.stock_symbols}
        self.sell_signals = {symbol: [] for symbol in self.stock_symbols}

        self.total_profit = 0
        self.trade_count = {"up": 0, "down": 0}

    def get_profit_rate(self):
        return self.trade_count["up"] / (
            self.trade_count["up"] + self.trade_count["down"] + 1e-6
        )

    def _get_portfolio_value(self):
        total_value = self.cash
        for symbol in self.stock_symbols:
            price = self.stock_data[symbol]["close"].iloc[self.current_step[symbol]]
            total_value += self.holdings[symbol] * price
        return total_value

    def _get_observation(self):
        obs = []
        for symbol_index, symbol in enumerate(self.stock_symbols):
            data = self.stock_data[symbol]
            current_step = self.current_step[symbol]
            start_index = max(0, current_step - self.window + 1)
            closing_prices = (
                data["close"]
                .iloc[start_index : current_step + 1]
                .values.astype(np.float32)
            )

            volume = (
                data["volume"]
                .iloc[start_index : current_step + 1]
                .values.astype(np.float32)
            )

            ma = (
                data["close_ma_100"]
                .iloc[start_index : current_step + 1]
                .values.astype(np.float32)
            )

            # Pad with the earliest price if not enough history
            if len(closing_prices) < self.window:
                padding = np.full(
                    self.window - len(closing_prices),
                    closing_prices[0] if len(closing_prices) > 0 else 0,
                    dtype=np.float32,
                )
                closing_prices = np.concatenate([padding, closing_prices])
                volume = np.concatenate([padding, volume])
                ma = np.concatenate([padding, ma])

            holding_status = np.array([self.holdings[symbol]], dtype=np.float32) * 100
            stock_obs = np.concatenate([closing_prices, volume, ma, holding_status])
            obs.append(stock_obs)
        return np.concatenate(obs)

    def _take_action(self, actions):
        reward = 0
        for i, action in enumerate(actions):
            symbol = self.stock_symbols[i]
            current_step = self.current_step[symbol]
            current_price = self.stock_data[symbol]["close"].iloc[current_step]
            date = self.stock_data[symbol]["date"].iloc[current_step]

            if action == 2:  # Buy
                # handle when agent tries to buy an asset with missing data
                if current_price == 0:
                    reward -= 0.5
                    continue
                if current_price <= self.cash:
                    self.holdings[symbol] += 1
                    self.purchase_price[symbol] += current_price
                    self.cash -= current_price
                    self.buy_signals[symbol].append(date)
                    reward += 2
            elif action == 0:  # Sell
                if self.holdings[symbol] > 0:
                    profit = (
                        current_price * self.holdings[symbol]
                        - self.purchase_price[symbol]
                    )
                    self.total_profit += profit
                    self.cash += current_price * self.holdings[symbol]

                    if profit > 0:
                        reward += 5 * profit / self.purchase_price[symbol]
                        self.trade_count["up"] += 1
                    else:
                        reward -= 1 * abs(profit) / self.purchase_price[symbol]
                        self.trade_count["down"] += 1

                    self.holdings[symbol] = 0
                    self.purchase_price[symbol] = 0
                    self.sell_signals[symbol].append(date)
                else:
                    reward -= 0.5

        return reward

    def step(self, actions):
        done = False
        observation = self._get_observation()
        reward = self._take_action(actions)

        for symbol in self.stock_symbols:
            self.current_step[symbol] += 1
            if self.current_step[symbol] >= len(self.stock_data[symbol]) - 1:
                done = True

        truncated = False
        info = {}
        info["buy_signals"] = self.buy_signals
        info["sell_signals"] = self.sell_signals
        info["profit_rate"] = self.get_profit_rate()
        info["cash"] = self._get_portfolio_value()
        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = {symbol: self.window - 1 for symbol in self.stock_symbols}
        self.holdings = {symbol: 0 for symbol in self.stock_symbols}
        self.purchase_price = {symbol: 0.0 for symbol in self.stock_symbols}
        self.buy_signals = {symbol: [] for symbol in self.stock_symbols}
        self.sell_signals = {symbol: [] for symbol in self.stock_symbols}
        self.total_profit = 0
        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self, mode="human"):
        if mode == "human":
            output = StringIO()
            print(
                f"Current Step: {next(iter(self.current_step.values()))}", file=output
            )
            print("Portfolio Holdings:", file=output)
            for i, symbol in enumerate(self.stock_symbols):
                holding_status = (
                    "Holding" if self.holdings[symbol] == 1 else "Not Holding"
                )
                cost_basis = (
                    f" (Bought @ {self.purchase_price[symbol]:.2f})"
                    if self.holdings[symbol] == 1
                    else ""
                )
                current_price = (
                    self.stock_data[symbol]["close"].iloc[self.current_step[symbol]]
                    if self.current_step[symbol] < len(self.stock_data[symbol])
                    else "N/A"
                )
                print(
                    f"  {symbol}: {holding_status}{cost_basis}, Current Price: {current_price:.2f}",
                    file=output,
                )
            print(output.getvalue())
            return output.getvalue()
        else:
            super().render()

    def close(self):
        pass
