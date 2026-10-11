import pandas as pd
from core.factor import Factor

class MomentumFactor(Factor):
    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        """
        12-minus-1 momentum by default: the return from ``lookback`` trading
        days ago to ``skip`` trading days ago. The most recent month is skipped
        because one-month returns tend to reverse (short-term reversal), which
        would otherwise work against the momentum signal.
        """
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback = lookback
        self.skip = skip

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Return the raw ``lookback``-minus-``skip`` return per (Ticker, Date).
        Rows without ``lookback`` days of history are NaN (warm-up), not 0.
        """
        self._validate(data, 'Close')
        close = data['Close'].sort_index().groupby(level='Ticker')
        momentum = close.shift(self.skip) / close.shift(self.lookback) - 1
        return momentum.reindex(data.index)
