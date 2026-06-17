import pandas as pd
import numpy as np
from core.factor import Factor

class VolumeFactor(Factor):
    name = "volume"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute volume ratio: current volume / 20-day average volume.
        Returns raw ratio (e.g., 1.5 means 50% above average) as a Series with same index as data.
        """
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['Ticker', 'Date']:
            raise ValueError("Data index must be MultiIndex with levels ['Ticker', 'Date']")
        
        if 'Volume' not in data.columns:
            raise ValueError("Data must contain 'Volume' column")
        
        result = pd.Series(index=data.index, dtype=float)
        for ticker, group in data.groupby(level='Ticker'):
            group = group.sort_index(level='Date')
            vol = group['Volume']
            avg_vol = vol.rolling(20).mean()
            # Avoid division by zero
            ratio = vol / avg_vol.where(avg_vol != 0, np.nan)
            # Fill NaN with 1 (ratio of 1 when insufficient data)
            ratio = ratio.fillna(1)
            result.loc[group.index] = ratio
        return result