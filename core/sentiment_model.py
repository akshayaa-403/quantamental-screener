from abc import ABC, abstractmethod
from typing import List, Dict

class SentimentModel(ABC):
    @abstractmethod
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        pass
    
    def batch_analyze(self, texts_by_ticker: Dict[str, List[str]]) -> Dict[str, Dict]:
        result = {}
        for ticker, texts in texts_by_ticker.items():
            result[ticker] = self.analyze(texts)
        return result