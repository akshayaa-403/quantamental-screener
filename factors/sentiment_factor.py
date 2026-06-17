import pandas as pd
from core.factor import Factor
import logging

logger = logging.getLogger(__name__)

class SentimentFactor(Factor):
    name = "sentiment"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute sentiment scores for each ticker on each date.
        Expects sentiment_scores as a dict of ticker -> (score or dict).
        If dict with 'score' key, uses that. If scalar, uses directly.
        The score is broadcast to all dates for that ticker (constant over time).
        For a real backtest, you would need historical sentiment per date.
        """
        sentiment_scores = kwargs.get('sentiment_scores', {})
        
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['Ticker', 'Date']:
            raise ValueError("Data index must be MultiIndex with levels ['Ticker', 'Date']")
        
        # Create a Series of zeros with the same index
        result = pd.Series(0.0, index=data.index)
        
        if not sentiment_scores:
            logger.warning("No sentiment scores provided. Returning zeros.")
            return result
        
        # For each ticker, extract score and broadcast to all its dates
        tickers = data.index.get_level_values('Ticker').unique()
        for ticker in tickers:
            score_data = sentiment_scores.get(ticker, {})
            if isinstance(score_data, dict):
                score = score_data.get('score', 0.0)
            elif isinstance(score_data, (int, float)):
                score = score_data
            else:
                score = 0.0
            # Assign to all rows for this ticker
            mask = data.index.get_level_values('Ticker') == ticker
            result.loc[mask] = score
        
        non_zero = (result != 0.0).sum()
        logger.info(f"Sentiment scores assigned: {non_zero}/{len(result)} non-zero values")
        return result