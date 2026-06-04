from core.factor import Factor
import pandas as pd

class MyFactor(Factor):
    name = "my_factor"
    weight = 0.0  # Disabled by default
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Placeholder factor - returns zero scores
        # Implement custom logic here
        scores = {}
        for ticker in data.index.get_level_values('Ticker').unique():
            # Calculate score for this ticker
            scores[ticker] = 0.0
        return pd.Series(scores)