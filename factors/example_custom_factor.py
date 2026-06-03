import pandas as pd
import numpy as np
from core.factor import Factor


class ExampleCustomFactor(Factor):
    """
    Example: Earnings Growth Factor
    
    This factor scores stocks based on their earnings growth rate.
    Higher growth yields higher scores.
    
    To use this factor:
    1. Copy this file to factors/custom_factor.py
    2. Rename the class appropriately
    3. Implement the compute method with your logic
    4. Add to the factors list in app.py or streamlit_app.py
    """
    
    name = "earnings_growth"  # Unique identifier for this factor
    weight = 0.15  # Default weight (can be overridden in config)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute raw scores for each ticker.
        
        Args:
            data: MultiIndex DataFrame with price/volume data
            **kwargs: Additional data like fundamentals, sentiment, etc.
            
        Returns:
            pd.Series: Scores indexed by ticker (raw, before normalization)
        """
        # Get fundamentals if available
        fundamentals = kwargs.get('fundamentals', {})
        
        scores = {}
        for ticker in data.index.get_level_values('Ticker').unique():
            # Example: use revenue growth from fundamentals
            ticker_fund = fundamentals.get(ticker, {})
            growth = ticker_fund.get('revenue_growth', 0)
            
            # Transform to a reasonable score range
            # Typical growth might be -0.5 to 0.5 (50% decline to 50% growth)
            score = np.clip(growth, -0.5, 0.5) / 0.5  # Scale to -1..1
            
            scores[ticker] = score
        
        return pd.Series(scores)
    
    def normalize(self, scores: pd.Series) -> pd.Series:
        """
        Optional: Override default normalization.
        For earnings growth, maybe you want different scaling.
        """
        # Use default min-max scaling
        return super().normalize(scores)


class ExampleCustomFactorV2(Factor):
    """
    Another Example: RSI-Based Momentum with Reversal
    
    This factor scores based on RSI but adds a reversal component
    for overbought/oversold conditions.
    """
    
    name = "rsi_momentum"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Use RSI with mean-reversion logic.
        RSI below 30 (oversold) gives positive score.
        RSI above 70 (overbought) gives negative score.
        """
        scores = {}
        
        for ticker in data.index.get_level_values('Ticker').unique():
            ticker_data = data.xs(ticker, level='Ticker').sort_index()
            
            if len(ticker_data) < 15 or 'RSI' not in ticker_data.columns:
                scores[ticker] = 0.0
                continue
            
            # Get latest RSI value
            current_rsi = ticker_data['RSI'].iloc[-1]
            
            # Score: higher when RSI is low (oversold)
            if current_rsi < 30:
                score = 1.0
            elif current_rsi > 70:
                score = -1.0
            else:
                # Linear mapping from 30-70 range to 1 to -1
                score = 1 - (current_rsi - 30) / 40 * 2
            
            scores[ticker] = score
        
        return pd.Series(scores)