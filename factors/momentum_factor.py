import pandas as pd
import numpy as np
from core.factor import Factor

class MomentumFactor(Factor):
    name = "momentum"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Ensure data has a MultiIndex with 'Ticker' level
        if not isinstance(data.index, pd.MultiIndex) or 'Ticker' not in data.index.names:
            # Fallback: assume data is a standard DataFrame with a 'Ticker' column
            if 'Ticker' in data.columns:
                grouped = data.groupby('Ticker')
            else:
                return pd.Series(dtype=float)
        else:
            grouped = data.groupby(level='Ticker')
        
        scores = {}
        for ticker, ticker_data in grouped:
            # Sort by date
            if isinstance(ticker_data.index, pd.MultiIndex):
                ticker_data = ticker_data.sort_index(level='Date')
            else:
                # If index is not MultiIndex, assume it has a Date column
                if 'Date' in ticker_data.columns:
                    ticker_data = ticker_data.sort_values('Date')
                else:
                    # Cannot sort, use as is
                    pass
            
            if len(ticker_data) < 21:
                scores[ticker] = 0.0
                continue
            
            # Get the 'Close' column
            if 'Close' not in ticker_data.columns:
                scores[ticker] = 0.0
                continue
            
            close = ticker_data['Close']
            
            # Convert to Series if it's a scalar (single value)
            if not isinstance(close, (pd.Series, pd.DataFrame)):
                close = pd.Series([close])
            
            # Ensure numeric and drop NaN
            try:
                close = pd.to_numeric(close, errors='coerce').dropna()
            except TypeError:
                # If conversion fails, treat as empty
                close = pd.Series(dtype=float)
            
            if len(close) < 21:
                scores[ticker] = 0.0
                continue
            
            # Rate of change: (current / 21 days ago) - 1
            # Use iloc if close is Series with integer index, else use values
            if isinstance(close, pd.Series):
                current = close.iloc[-1]
                past = close.iloc[-21]
            else:
                # Fallback to numpy array
                current = close[-1]
                past = close[-21]
            
            roc = (current / past - 1) if past != 0 else 0.0
            # Clip to [-1, 1] assuming max 20% move (normalization step)
            norm_roc = np.clip(roc / 0.2, -1, 1)
            scores[ticker] = norm_roc
        
        return pd.Series(scores)