import pandas as pd
import numpy as np
from core.factor import Factor

class VolatilityFactor(Factor):
    name = "volatility"
    
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
                
                close_series = ticker_data['Close'].dropna()
                if len(close_series) < 20:
                    scores[ticker] = 0.0
                    continue
                
                # Calculate daily returns
                returns = close_series.pct_change().dropna()
                if len(returns) < 2:
                    scores[ticker] = 0.0
                    continue
                
                # Annualized volatility
                vol = returns.std() * np.sqrt(252)
                # Ensure scalar
                vol = float(vol) if not pd.isna(vol) else 0.3
                
                # Normalize: typical vol range 0.2 - 0.5, lower vol gets higher score
                norm_vol = 1 - np.clip((vol - 0.2) / 0.3, 0, 1)
                norm_vol = norm_vol * 2 - 1  # map to [-1, 1]
                scores[ticker] = norm_vol
            except Exception:
                scores[ticker] = 0.0
        
        return pd.Series(scores)