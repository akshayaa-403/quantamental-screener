from abc import ABC, abstractmethod
import pandas as pd

class Factor(ABC):
    name: str = "base_factor"
    weight: float = 1.0
    # A required factor must have a value for a row to be rankable at all, so
    # rows still inside a factor's warm-up window get no composite score. An
    # optional factor (sentiment) is simply treated as neutral when missing.
    required: bool = True
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        pass

    @staticmethod
    def _validate(data: pd.DataFrame, column: str = None) -> None:
        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != ['Ticker', 'Date']:
            raise ValueError("Data index must be MultiIndex with levels ['Ticker', 'Date']")
        if column is not None and column not in data.columns:
            raise ValueError(f"Data must contain '{column}' column")
