import pandas as pd
import numpy as np
from core.factor import Factor

class VolumeFactor(Factor):
    name = "volume"

    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Signed relative volume: (volume / ``window``-day average volume) times
        the sign of that day's return. Heavy buying scores high, heavy selling
        scores low; raw volume alone would rank a panic sell-off as highly as
        buying enthusiasm. Rows without a full window are NaN (warm-up).
        """
        self._validate(data, 'Close')
        self._validate(data, 'Volume')
        ordered = data.sort_index()
        by_ticker = ordered.groupby(level='Ticker')
        avg_vol = by_ticker['Volume'].transform(lambda v: v.rolling(self.window).mean())
        ratio = ordered['Volume'] / avg_vol.where(avg_vol != 0)
        direction = np.sign(by_ticker['Close'].pct_change())
        return (ratio * direction).reindex(data.index)
