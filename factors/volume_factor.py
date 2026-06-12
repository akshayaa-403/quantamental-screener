import pandas as pd
import numpy as np
from core.factor import Factor

class VolumeFactor(Factor):
    name = "volume"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        scores = {}
        # Get unique tickers regardless of index structure
        if isinstance(data.index, pd.MultiIndex):
            tickers = data.index.get_level_values('Ticker').unique()
        else:
            tickers = data['Ticker'].unique() if 'Ticker' in data.columns else []
        
        for ticker in tickers:
            try:
                # Extract data for this ticker
                if isinstance(data.index, pd.MultiIndex):
                    ticker_data = data.xs(ticker, level='Ticker').sort_index()
                else:
                    ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
                
                if len(ticker_data) < 20:
                    scores[ticker] = 0.0
                    continue
                
                volume_series = ticker_data['Volume'].dropna()
                if len(volume_series) < 20:
                    scores[ticker] = 0.0
                    continue
                
                # Calculate average volume over 20 days (last value)
                avg_volume = volume_series.rolling(20).mean().iloc[-1]
                curr_volume = volume_series.iloc[-1]
                
                # Ensure scalars (convert from pandas types)
                avg_volume = float(avg_volume) if not pd.isna(avg_volume) else 0.0
                curr_volume = float(curr_volume) if not pd.isna(curr_volume) else 0.0
                
                if avg_volume > 0:
                    vol_ratio = curr_volume / avg_volume
                else:
                    vol_ratio = 1.0
                
                # Normalize: cap at 3x average volume, map to [-1, 1]
                norm_ratio = np.clip(vol_ratio / 3.0, 0, 1) * 2 - 1
                scores[ticker] = norm_ratio
            except Exception:
                scores[ticker] = 0.0
        
        return pd.Series(scores)