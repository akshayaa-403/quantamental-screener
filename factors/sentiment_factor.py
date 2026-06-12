import pandas as pd
from core.factor import Factor
import logging

logger = logging.getLogger(__name__)

class SentimentFactor(Factor):
    name = "sentiment"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        sentiment_scores = kwargs.get('sentiment_scores', {})
        
        # Extract unique tickers from data (handles both MultiIndex and flat DataFrame)
        if isinstance(data.index, pd.MultiIndex) and 'Ticker' in data.index.names:
            tickers = data.index.get_level_values('Ticker').unique()
        elif 'Ticker' in data.columns:
            tickers = data['Ticker'].unique()
        else:
            logger.warning("Could not extract tickers from data. Returning zeros.")
            return pd.Series(dtype=float)
        
        if not sentiment_scores:
            logger.warning("No sentiment scores provided. Returning zeros for all tickers.")
            return pd.Series(0.0, index=tickers)
        
        scores = {}
        for t in tickers:
            score_data = sentiment_scores.get(t, {})
            if isinstance(score_data, dict):
                score = score_data.get('score', 0.0)
            elif isinstance(score_data, (int, float)):
                score = score_data
            else:
                score = 0.0
            scores[t] = score
        
        # Log non-zero sentiment counts
        non_zero = sum(1 for v in scores.values() if v != 0.0)
        logger.info(f"Sentiment scores computed for {non_zero}/{len(tickers)} tickers")
        
        return pd.Series(scores)