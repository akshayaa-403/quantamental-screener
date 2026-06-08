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
        if isinstance(scores, pd.DataFrame):
            scores = scores.iloc[:, 0]
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)
        
        values = scores.values
        mask = ~pd.isna(values)
        clean = values[mask]
        if len(clean) == 0:
            return pd.Series(0.0, index=scores.index)
        
        min_val = np.min(clean)
        max_val = np.max(clean)
        if max_val == min_val:
            return pd.Series(0.0, index=scores.index)
        
        normalized = (values - min_val) / (max_val - min_val) * 2 - 1
        return pd.Series(normalized, index=scores.index)