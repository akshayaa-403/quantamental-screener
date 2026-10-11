from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from core.factor import Factor

class ScoringEngine:
    def __init__(self, factors: List[Factor], weights: Dict[str, float] = None):
        self.factors = factors
        # The engine owns a private copy of the weights. Factor objects are
        # shared (the CLI and the dashboard reuse them between the backtest and
        # the live ranking), so writing to ``factor.weight`` would leak one
        # run's weights into the next.
        weights = weights or {}
        self.weights = {f.name: float(weights.get(f.name, f.weight)) for f in factors}

    def compute(self, data: pd.DataFrame, sentiment_scores: Optional[Dict] = None,
                weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Score every (Ticker, Date) row.

        ``weights`` overrides this engine's weights for this call only. A row
        whose required factors (those with a non-zero weight) are still in
        their warm-up window gets a NaN composite and no rank, so it is never
        mistaken for an average stock.
        """
        active = {**self.weights, **(weights or {})}

        factor_scores = {}
        warming_up = pd.Series(False, index=data.index)
        for factor in self.factors:
            kwargs = {}
            if factor.name == 'sentiment':
                kwargs['sentiment_scores'] = sentiment_scores or {}
            raw = factor.compute(data, **kwargs)

            if not isinstance(raw.index, pd.MultiIndex) or raw.index.nlevels != 2:
                raise ValueError(f"Factor {factor.name} returned a Series with invalid index. "
                                 f"Expected MultiIndex of 2 levels, got {raw.index}")
            if list(raw.index.names) != ['Ticker', 'Date']:
                raw = raw.rename_axis(['Ticker', 'Date'])

            if factor.required and active.get(factor.name, 0.0) != 0.0:
                warming_up |= raw.reindex(data.index).isna()

            # Standardize within each date, never across dates: scaling by the
            # whole panel would let future values move past scores.
            factor_scores[factor.name] = raw.groupby(level='Date').transform(self._standardize_series)

        combined = pd.DataFrame(factor_scores)
        if factor_scores:
            combined['composite'] = sum(
                combined[name] * active.get(name, 0.0) for name in factor_scores
            )
        else:
            combined['composite'] = 0.0
        combined.loc[warming_up.reindex(combined.index).to_numpy(), 'composite'] = np.nan

        # Rank per date
        combined['rank'] = combined.groupby(level='Date')['composite'].rank(ascending=False)
        return combined

    @staticmethod
    def _standardize_series(s: pd.Series) -> pd.Series:
        """Cross-sectional z-score standardization: (x - mean) / std.

        Standardization is the standard choice for multi-factor models: unlike
        min-max scaling it is not distorted by a single outlier stock, and it
        puts every factor on a common mean-0 / unit-variance scale so the
        configured weights are directly comparable. Scores are clipped to +/-3
        to cap the influence of extreme outliers. Missing values map to 0 (the
        cross-sectional average), as does a degenerate (constant or empty)
        cross-section.
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
