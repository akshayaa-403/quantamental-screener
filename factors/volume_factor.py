import pandas as pd
import numpy as np
from core.factor import Factor

class VolumeFactor(Factor):
    name = "volume"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        latest = data.groupby(level='Ticker').last()
        scores = {}
        for ticker in latest.index:
            ticker_data = data.xs(ticker, level='Ticker').sort_index()
            if len(ticker_data) < 20:
                scores[ticker] = 0.0
                continue
            avg_volume = ticker_data['Volume'].rolling(20).mean().iloc[-1]
            curr_volume = ticker_data['Volume'].iloc[-1]
            vol_ratio = curr_volume / avg_volume if avg_volume > 0 else 1.0
            norm_ratio = np.clip(vol_ratio / 3.0, 0, 1) * 2 - 1
            scores[ticker] = norm_ratio
        return pd.Series(scores)