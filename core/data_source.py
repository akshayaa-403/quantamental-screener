from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import pandas as pd

class DataSource(ABC):
    @abstractmethod
    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Return MultiIndex DataFrame (ticker, date) with columns: Open, High, Low, Close, Volume.
        """
        pass
    
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Dict:
        """
        Return dict with keys like 'market_cap', 'pe_ratio', 'revenue_growth'.
        Optional: can return empty dict if not available.
        """
        pass
    
    @abstractmethod
    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        """
        Return list of news headlines for the given ticker.
        """
        pass