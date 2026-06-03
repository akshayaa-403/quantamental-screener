import pandas as pd
import numpy as np
from core.factor import Factor

class MomentumFactor(Factor):
    name = "momentum"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        latest = data.groupby(level='Ticker').last()
        scores = {}
        
        for ticker, row in latest.iterrows():
            ticker_data = data.xs(ticker, level='Ticker').sort_index()
            if len(ticker_data) < 20:
                scores[ticker] = 0.0
                continue
            close = ticker_data['Close']
            roc = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
            norm_roc = np.clip(roc / 0.2, -1, 1)
            scores[ticker] = norm_roc
        return pd.Series(scores)