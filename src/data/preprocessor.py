import numpy as np


def normalize_prices(data, column="Close"):
    """Normalize price data."""
    prices = data[column].values
    mean = prices.mean()
    std = prices.std()
    return (prices - mean) / std, mean, std
