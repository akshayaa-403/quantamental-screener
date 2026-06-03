from core.factor import Factor
import pandas as pd

class MyFactor(Factor):
    name = "my_factor"
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Your scoring logic here
        scores = {}
        for ticker in data.index.get_level_values('Ticker').unique():
            # Calculate score for this ticker
            scores[ticker] = some_value
        return pd.Series(scores)