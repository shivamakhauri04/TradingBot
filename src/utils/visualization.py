from plotly import tools
from plotly.graph_objs import Candlestick, Figure
from plotly.offline import iplot


def plot_train_test(train, test, date_split):
    """Plot training and test data as candlestick charts."""
    data = [
        Candlestick(
            x=train.index,
            open=train["Open"],
            high=train["High"],
            low=train["Low"],
            close=train["Close"],
            name="train",
        ),
        Candlestick(
            x=test.index,
            open=test["Open"],
            high=test["High"],
            low=test["Low"],
            close=test["Close"],
            name="test",
        ),
    ]
    layout = {
        "shapes": [
            {
                "x0": date_split,
                "x1": date_split,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "rgb(0,0,0)", "width": 1},
            }
        ],
        "annotations": [
            {
                "x": date_split,
                "y": 1.0,
                "xref": "x",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
                "text": " test data",
            },
            {
                "x": date_split,
                "y": 1.0,
                "xref": "x",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "right",
                "text": "train data ",
            },
        ],
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)


def plot_rewards(total_rewards, title="Training Rewards"):
    """Plot training rewards over episodes."""
    import plotly.graph_objs as go
    from plotly.offline import iplot

    trace = go.Scatter(y=total_rewards, mode="lines", name="Rewards")
    layout = go.Layout(title=title, xaxis=dict(title="Episode"), yaxis=dict(title="Reward"))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
