import numpy as np
import pandas as pd
from core.factor import Factor
import logging

logger = logging.getLogger(__name__)

class SentimentFactor(Factor):
    name = "sentiment"
    # Missing sentiment is treated as neutral, not as a reason to drop a row.
    required = False
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Broadcast each ticker's sentiment score to all of its dates.

        Expects ``sentiment_scores`` as a dict of ticker -> (score or dict with
        a 'score' key). Tickers with no score are NaN, not 0: after
        cross-sectional z-scoring a raw 0 is not neutral (it sits below average
        whenever the covered names are net positive), whereas NaN is filled
        with a z-score of exactly 0.

        The scores are a snapshot of *current* news, so this factor must only
        be used for the live ranking, never in a backtest.
        """
        sentiment_scores = kwargs.get('sentiment_scores') or {}
        self._validate(data)

        def _score(value):
            if isinstance(value, dict):
                value = value.get('score')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return np.nan

        per_ticker = {t: _score(v) for t, v in sentiment_scores.items()}
        tickers = data.index.get_level_values('Ticker')
        result = pd.Series(tickers.map(lambda t: per_ticker.get(t, np.nan)), index=data.index, dtype=float)

        if per_ticker:
            covered = result.groupby(level='Ticker').first().notna().sum()
            logger.info(f"Sentiment coverage: {covered}/{tickers.nunique()} tickers")
        return result
