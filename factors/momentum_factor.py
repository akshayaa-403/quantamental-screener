import pandas as pd
import numpy as np
from core.factor import Factor

class MomentumFactor(Factor):
    name = "momentum"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute momentum as the 21-day rate of change (ROC) for each ticker.
        Returns raw ROC values (e.g., 0.05 for 5% gain) as a Series with the same index as data.
        """
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['Ticker', 'Date']:
            raise ValueError("Data index must be MultiIndex with levels ['Ticker', 'Date']")
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        result = pd.Series(index=data.index, dtype=float)
        for ticker, group in data.groupby(level='Ticker'):
            group = group.sort_index(level='Date')
            close = group['Close']
            # 21-day rate of change
            roc = close / close.shift(21) - 1
            # Fill NaN with 0 for early dates
            roc = roc.fillna(0)
            result.loc[group.index] = roc
        return result