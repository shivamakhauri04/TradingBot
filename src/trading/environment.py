import numpy as np


class TradingEnvironment:
    """Gym-compatible trading environment for DQN training.

    Actions:
        0: Hold (do nothing)
        1: Buy
        2: Sell all positions
    """

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history

    def step(self, act):
        reward = 0

        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]["Close"])
        elif act == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += self.data.iloc[self.t, :]["Close"] - p
                reward += profits
                self.profits += profits
                self.positions = []

        self.t += 1

        self.position_value = 0
        for p in self.positions:
            self.position_value += self.data.iloc[self.t, :]["Close"] - p
        self.history.pop(0)
        self.history.append(
            self.data.iloc[self.t, :]["Close"] - self.data.iloc[(self.t - 1), :]["Close"]
        )
        if self.t == len(self.data) - 1:
            self.done = True

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done
