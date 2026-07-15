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
                # Group by the second level (date) and standardize cross-sectionally
                normalized = raw.groupby(level=1).transform(self._standardize_series)
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
    def _standardize_series(s: pd.Series) -> pd.Series:
        """Cross-sectional z-score standardization: (x - mean) / std.

        Standardization is the standard choice for multi-factor models: unlike
        min-max scaling it is not distorted by a single outlier stock, and it
        puts every factor on a common mean-0 / unit-variance scale so the
        configured weights are directly comparable. Scores are clipped to +/-3
        to cap the influence of extreme outliers. A degenerate (constant or
        empty) cross-section maps to 0.
        """
        clean = s.dropna()
        if clean.empty:
            return pd.Series(0.0, index=s.index)
        mean = clean.mean()
        std = clean.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        z = (s - mean) / std
        return z.clip(-3, 3).fillna(0.0)