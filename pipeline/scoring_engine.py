from typing import List, Dict, Optional
import pandas as pd
from core.factor import Factor

class ScoringEngine:
    def __init__(self, factors: List[Factor], weights: Dict[str, float] = None):
        self.factors = factors
        if weights:
            for factor in self.factors:
                if factor.name in weights:
                    factor.weight = weights[factor.name]

    def compute(self, data: pd.DataFrame, sentiment_scores: Optional[Dict] = None,
                cross_sectional_normalize: bool = False) -> pd.DataFrame:
        
        factor_scores = {}
        for factor in self.factors:
            kwargs = {}
            if factor.name == 'sentiment':
                kwargs['sentiment_scores'] = sentiment_scores or {}
            raw = factor.compute(data, **kwargs)
            
            if not isinstance(raw.index, pd.MultiIndex) or raw.index.nlevels != 2:
                raise ValueError(f"Factor {factor.name} returned a Series with invalid index. "
                                 f"Expected MultiIndex of 2 levels, got {raw.index}")

            if cross_sectional_normalize:
                # Normalize per date (cross‑sectional)
                
                if raw.index.names[0] != 'Ticker' or raw.index.names[1] != 'Date':
                    raw.index.set_names(['Ticker', 'Date'], inplace=True)
                # Group by the second level (date) and apply normalization
                normalized = raw.groupby(level=1).transform(self._normalize_series)
            else:
                normalized = factor.normalize(raw)

            factor_scores[factor.name] = normalized

        combined = pd.DataFrame(factor_scores)
        numeric_cols = [col for col in combined.columns if col in [f.name for f in self.factors]]
        if numeric_cols:
            combined['composite'] = sum(
                combined[col] * next(f.weight for f in self.factors if f.name == col)
                for col in numeric_cols
            )
        else:
            combined['composite'] = 0.0

        # Rank per date
        combined['rank'] = combined.groupby(level=1)['composite'].rank(ascending=False)
        return combined

    @staticmethod
    def _normalize_series(s: pd.Series) -> pd.Series:
        
        clean = s.dropna()
        if clean.empty:
            return pd.Series(0.0, index=s.index)
        min_val = clean.min()
        max_val = clean.max()
        if max_val == min_val:
            return pd.Series(0.0, index=s.index)
        return (s - min_val) / (max_val - min_val) * 2 - 1