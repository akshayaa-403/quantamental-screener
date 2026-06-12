from typing import List, Dict
import pandas as pd
from core.factor import Factor

class ScoringEngine:
    def __init__(self, factors: List[Factor], weights: Dict[str, float] = None):
        self.factors = factors
        if weights:
            for factor in self.factors:
                if factor.name in weights:
                    factor.weight = weights[factor.name]
    
    def compute(self, data: pd.DataFrame, sentiment_scores: Dict = None) -> pd.DataFrame:
        factor_scores = {}
        for factor in self.factors:
            kwargs = {}
            if factor.name == 'sentiment':
                kwargs['sentiment_scores'] = sentiment_scores or {}
            raw = factor.compute(data, **kwargs)
            normalized = factor.normalize(raw)
            factor_scores[factor.name] = normalized
        
        # Combine scores
        combined = pd.DataFrame(factor_scores)
        # Use only numeric columns that match factor names
        numeric_cols = [col for col in combined.columns if col in [f.name for f in self.factors]]
        if numeric_cols:
            combined['composite'] = sum(
                combined[col] * next(f.weight for f in self.factors if f.name == col)
                for col in numeric_cols
            )
        else:
            combined['composite'] = 0.0
        combined['rank'] = combined['composite'].rank(ascending=False)
        return combined