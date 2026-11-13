import pandas as pd
import numpy as np
from core.factor import Factor

class VolumeFactor(Factor):
    name = "volume"

    def __init__(self, window: int = 20, smooth: int = 10):
        """
        ``window``: days in the average volume that each day is compared with.
        ``smooth``: days of signed relative volume averaged into the signal.
        """
        if smooth < 1:
            raise ValueError("smooth must be at least 1")
        self.window = window
        self.smooth = smooth

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Smoothed signed relative volume: each day's volume / ``window``-day
        average volume, times the sign of that day's return, averaged over the
        last ``smooth`` days. It measures sustained buying or selling pressure
        (accumulation vs distribution). A single day of it is mostly noise and
        overlaps with the one-day reversal effect, hence the smoothing. Rows
        without a full window are NaN (warm-up).
        """
        self._validate(data, 'Close')
        self._validate(data, 'Volume')
        ordered = data.sort_index()
        by_ticker = ordered.groupby(level='Ticker')
        avg_vol = by_ticker['Volume'].transform(lambda v: v.rolling(self.window).mean())
        ratio = ordered['Volume'] / avg_vol.where(avg_vol != 0)
        direction = np.sign(by_ticker['Close'].pct_change())
        signed = ratio * direction
        smoothed = signed.groupby(level='Ticker').transform(lambda s: s.rolling(self.smooth).mean())
        return smoothed.reindex(data.index)
