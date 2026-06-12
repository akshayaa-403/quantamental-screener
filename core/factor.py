from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Factor(ABC):
    name: str = "base_factor"
    weight: float = 1.0
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        pass
    
    def normalize(self, scores: pd.Series) -> pd.Series:
        # Ensure scores is a Series
        if isinstance(scores, pd.DataFrame):
            if scores.shape[1] == 0:
                return pd.Series(0.0, index=scores.index)
            scores = scores.iloc[:, 0]
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)
        
        # Drop NaN values for min/max calculation
        clean = scores.dropna()
        if len(clean) == 0:
            return pd.Series(0.0, index=scores.index)
        
        # Use pandas min/max for robustness (avoids ambiguous truth errors)
        min_val = clean.min()
        max_val = clean.max()
        if max_val == min_val:
            return pd.Series(0.0, index=scores.index)
        
        # Normalize to [-1, 1]
        normalized = (scores - min_val) / (max_val - min_val) * 2 - 1
        return normalized.fillna(0.0)