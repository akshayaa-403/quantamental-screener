import pandas as pd
import numpy as np
from core.factor import Factor

class VolatilityFactor(Factor):
    name = "volatility"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        scores = {}
        for ticker in data.index.get_level_values('Ticker').unique():
            ticker_data = data.xs(ticker, level='Ticker').sort_index()
            if len(ticker_data) < 20:
                scores[ticker] = 0.0
                continue
            returns = ticker_data['Close'].pct_change().dropna()
            vol = returns.std() * np.sqrt(252)  # annualized
            # Normalize: assume typical vol 0.2-0.4 range
            norm_vol = 1 - np.clip((vol - 0.2) / 0.3, 0, 1)
            norm_vol = norm_vol * 2 - 1  # map to -1..1
            scores[ticker] = norm_vol
        return pd.Series(scores)