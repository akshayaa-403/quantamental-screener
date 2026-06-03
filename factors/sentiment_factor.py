import pandas as pd
from core.factor import Factor

class SentimentFactor(Factor):
    name = "sentiment"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        sentiment_scores = kwargs.get('sentiment_scores', {})
        if not sentiment_scores:
            return pd.Series(0.0, index=data.index.get_level_values('Ticker').unique())
        series = pd.Series({t: sentiment_scores.get(t, {}).get('score', 0.0) 
                            for t in data.index.get_level_values('Ticker').unique()})
        return series