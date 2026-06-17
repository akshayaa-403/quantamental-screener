import pandas as pd
import numpy as np
from core.factor import Factor

class VolatilityFactor(Factor):
    name = "volatility"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute annualized volatility (standard deviation of daily returns * sqrt(252)).
        Returns raw volatility values (e.g., 0.25 for 25%) as a Series with the same index as data.
        """
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['Ticker', 'Date']:
            raise ValueError("Data index must be MultiIndex with levels ['Ticker', 'Date']")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        result = pd.Series(index=data.index, dtype=float)
        for ticker, group in data.groupby(level='Ticker'):
            group = group.sort_index(level='Date')
            close = group['Close']
            # Daily returns
            returns = close.pct_change()
            # Rolling 20-day volatility (annualized)
            vol = returns.rolling(20).std() * np.sqrt(252)
            # Fill NaN with a default (e.g., 0.3) for early periods
            vol = vol.fillna(0.3)
            result.loc[group.index] = vol
        return result