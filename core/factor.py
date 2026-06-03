from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict

class Factor(ABC):
    name: str = "base_factor"
    weight: float = 1.0
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        pass
    
    def normalize(self, scores: pd.Series) -> pd.Series:
        min_val = scores.min()
        max_val = scores.max()
        if max_val == min_val:
            return pd.Series(0.0, index=scores.index)
        return (scores - min_val) / (max_val - min_val) * 2 - 1