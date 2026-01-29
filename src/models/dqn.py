import copy
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""

    def __init__(self, obs_len, hidden_size, actions_n):
        super(QNetwork, self).__init__()
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions_n),
        )

    def forward(self, x):
        return self.fc_val(x)
