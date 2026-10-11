import pandas as pd
import numpy as np
from core.factor import Factor

class VolatilityFactor(Factor):
    name = "volatility"

    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute a low-volatility factor score.

        We measure annualized volatility (std of daily returns * sqrt(252)) and
        return its *negative*, so that lower-volatility (more stable) stocks get
        a higher score. This matches the low-volatility anomaly and keeps the
        factor consistent with the scoring engine, which treats larger values as
        better. Returned as a Series with the same index as ``data``; rows
        without a full ``window`` of returns are NaN (warm-up).
        """
        self._validate(data, 'Close')
        returns = data['Close'].sort_index().groupby(level='Ticker').pct_change()
        vol = (
            returns.groupby(level='Ticker')
            .rolling(self.window)
            .std()
            .droplevel(0)
            * np.sqrt(252)
        )
        return -vol.reindex(data.index)
