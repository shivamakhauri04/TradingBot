import pandas as pd


def load_stock_data(filepath, date_col="Date"):
    """Load stock data from CSV file."""
    data = pd.read_csv(filepath)
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    return data


def split_data(data, split_date="2016-01-01"):
    """Split data into train and test sets by date."""
    train = data[:split_date]
    test = data[split_date:]
    return train, test
