"""Unit tests for the factor implementations.

Factors operate on a ``(Ticker, Date)`` MultiIndex and return one value per row
(the score for that ticker on that date), so the tests assert on the full index
rather than on a single value per ticker.
"""

import numpy as np
import pandas as pd
import pytest

from factors import MomentumFactor, SentimentFactor, VolatilityFactor, VolumeFactor
from factors.my_factor import MyFactor


def _single_ticker(closes, ticker="AAA", volume=1_000_000):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    idx = pd.MultiIndex.from_product([[ticker], dates], names=["Ticker", "Date"])
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float), "Volume": volume}, index=idx)


class TestMomentumFactor:
    def test_returns_series_over_full_index(self, price_data):
        data = price_data(["AAPL", "MSFT"], periods=40)
        result = MomentumFactor().compute(data)
        assert isinstance(result, pd.Series)
        assert result.index.equals(data.index)
        assert not result.isna().any()

    def test_rejects_non_multiindex(self):
        with pytest.raises(ValueError):
            MomentumFactor().compute(pd.DataFrame({"Close": [1, 2, 3]}))

    def test_positive_momentum_for_rising_prices(self):
        data = _single_ticker(np.arange(100, 130))  # strictly increasing
        result = MomentumFactor().compute(data)
        assert result.iloc[-1] > 0


class TestNormalize:
    def test_maps_extremes_to_unit_range(self):
        norm = MomentumFactor().normalize(pd.Series([-5.0, 0.0, 3.0, 10.0]))
        assert norm.min() >= -1 and norm.max() <= 1
        assert norm.iloc[0] == pytest.approx(-1.0)
        assert norm.iloc[-1] == pytest.approx(1.0)

    def test_constant_series_is_zero(self):
        norm = MomentumFactor().normalize(pd.Series([2.0, 2.0, 2.0]))
        assert (norm == 0.0).all()


class TestSentimentFactor:
    def test_broadcasts_scores_across_dates(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["Ticker", "Date"])
        data = pd.DataFrame(index=idx)
        scores = {"AAPL": {"score": 0.8}, "MSFT": {"score": -0.2}}

        result = SentimentFactor().compute(data, sentiment_scores=scores)

        assert (result.xs("AAPL", level="Ticker") == 0.8).all()
        assert (result.xs("MSFT", level="Ticker") == -0.2).all()

    def test_missing_scores_default_to_zero(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        idx = pd.MultiIndex.from_product([["AAPL"], dates], names=["Ticker", "Date"])
        result = SentimentFactor().compute(pd.DataFrame(index=idx))
        assert (result == 0.0).all()


class TestVolumeFactor:
    def test_returns_series_over_full_index(self, price_data):
        data = price_data(["AAPL"], periods=40)
        result = VolumeFactor().compute(data)
        assert isinstance(result, pd.Series)
        assert result.index.equals(data.index)
        assert not result.isna().any()


class TestVolatilityFactor:
    def test_lower_volatility_scores_higher(self):
        rng = np.random.default_rng(123)
        low = 100 + np.cumsum(rng.normal(0, 0.2, size=40))
        high = 100 + np.cumsum(rng.normal(0, 3.0, size=40))
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        frames = []
        for ticker, close in [("LOW_VOL", low), ("HIGH_VOL", high)]:
            df = pd.DataFrame({"Close": close, "Volume": 1_000_000})
            df["Ticker"] = ticker
            df["Date"] = dates
            frames.append(df)
        data = pd.concat(frames, ignore_index=True).set_index(["Ticker", "Date"]).sort_index()

        result = VolatilityFactor().compute(data)

        low_score = result.xs("LOW_VOL", level="Ticker").mean()
        high_score = result.xs("HIGH_VOL", level="Ticker").mean()
        assert low_score > high_score


class TestMyFactor:
    def test_disabled_by_default(self):
        assert MyFactor().weight == 0.0
