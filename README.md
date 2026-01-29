# TradingBot

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.3+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Stock price prediction and automated trading using Deep Q-Network (DQN) reinforcement learning. Trains an agent to make buy/sell/hold decisions on historical stock data.

## Architecture

This project implements a **Deep Q-Network (DQN)** agent for automated stock trading. The key components are:

### Trading Environment
A custom gym-compatible environment that simulates stock trading. At each time step, the agent observes the current portfolio value and recent price change history, then selects one of three actions:
- **Hold** (action 0): Do nothing
- **Buy** (action 1): Purchase stock at the current price
- **Sell** (action 2): Sell all held positions

Rewards are clipped to +1 (profitable trade), -1 (unprofitable trade or selling with no positions), or 0 (hold/buy).

### Q-Network
A fully connected neural network that maps observations to Q-values for each action. The network consists of three linear layers with ReLU activations:
```
Input (obs_len) -> Hidden (100) -> Hidden (100) -> Output (3 actions)
```

### Training Strategy
- **Experience Replay**: Stores past transitions in a replay buffer and samples random mini-batches for training, breaking temporal correlations
- **Target Network**: A separate target Q-network is periodically updated with the weights of the online network, stabilizing training
- **Epsilon-Greedy Exploration**: Starts with random actions (epsilon=1.0) and gradually shifts to greedy policy (epsilon=0.1) as the agent learns

## Results

The DQN agent was trained on Google (GOOG) stock data from 2014-2016 and tested on 2016-2017 data. The agent achieved **test profits of $3,841.44** on the held-out test period.

## Project Structure

```
TradingBot/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── dqn.py                # Deep Q-Network model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py            # Data loading utilities
│   │   └── preprocessor.py       # Feature engineering
│   ├── trading/
│   │   ├── __init__.py
│   │   └── environment.py        # Trading environment
│   └── utils/
│       ├── __init__.py
│       └── visualization.py      # Plotting utilities
├── notebooks/
│   └── 01_original_dqn.ipynb     # Original DQN notebook
├── configs/
│   └── default.yaml              # Training hyperparameters
├── scripts/                      # Training and evaluation scripts
├── tests/                        # Unit tests
├── experiments/                  # Experiment logs (gitignored)
├── checkpoints/                  # Model checkpoints (gitignored)
├── data/                         # Data files (gitignored)
├── assets/                       # Static assets
├── docs/                         # Documentation
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Getting Started

### Installation

```bash
git clone https://github.com/shivamakhauri/TradingBot.git
cd TradingBot
pip install -r requirements.txt
```

Or install as a package with development dependencies:

```bash
pip install -e ".[dev]"
```

### Usage

#### Jupyter Notebook

The original interactive notebook is available at `notebooks/01_original_dqn.ipynb`. Launch it with:

```bash
jupyter notebook notebooks/01_original_dqn.ipynb
```

#### Modular Code

You can also use the extracted modules directly:

```python
from src.data.fetcher import load_stock_data, split_data
from src.trading.environment import TradingEnvironment
from src.models.dqn import QNetwork

# Load and split data
data = load_stock_data("Data/Stocks/goog.us.txt")
train, test = split_data(data, split_date="2016-01-01")

# Create environment
env = TradingEnvironment(train, history_t=90)

# Initialize Q-Network
obs_len = env.history_t + 1
model = QNetwork(obs_len=obs_len, hidden_size=100, actions_n=3)
```

## Configuration

Training hyperparameters are defined in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Number of training epochs |
| `memory_size` | 200 | Experience replay buffer size |
| `batch_size` | 50 | Training batch size |
| `gamma` | 0.97 | Discount factor |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.1 | Minimum exploration rate |
| `hidden_size` | 100 | Q-Network hidden layer size |
| `history_t` | 90 | Price history window length |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
