"""Unit tests for the factor implementations.

Factors operate on a ``(Ticker, Date)`` MultiIndex and return one value per row
(the score for that ticker on that date), so the tests assert on the full index
rather than on a single value per ticker.
"""

import numpy as np
import pandas as pd
import pytest

from factors import MomentumFactor, SentimentFactor, VolatilityFactor, VolumeFactor


def _single_ticker(closes, ticker="AAA", volume=1_000_000):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    idx = pd.MultiIndex.from_product([[ticker], dates], names=["Ticker", "Date"])
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float), "Volume": volume}, index=idx)


class TestMomentumFactor:
    def test_returns_series_over_full_index(self, price_data):
        data = price_data(["AAPL", "MSFT"], periods=300)
        result = MomentumFactor().compute(data)
        assert isinstance(result, pd.Series)
        assert result.index.equals(data.index)
        # Warm-up rows are NaN, not a made-up 0; everything after is filled.
        for ticker in ["AAPL", "MSFT"]:
            series = result.xs(ticker, level="Ticker")
            assert series.iloc[:252].isna().all()
            assert series.iloc[252:].notna().all()

    def test_skips_most_recent_month(self):
        # A crash inside the skipped month must not move the 12-1 signal.
        closes = np.linspace(100, 200, 300)
        crashed = closes.copy()
        crashed[-10:] = 50
        base = MomentumFactor().compute(_single_ticker(closes)).iloc[-1]
        assert MomentumFactor().compute(_single_ticker(crashed)).iloc[-1] == pytest.approx(base)

    def test_lookback_minus_skip_return(self):
        closes = np.arange(1, 301, dtype=float)
        result = MomentumFactor(lookback=252, skip=21).compute(_single_ticker(closes))
        # Last row (index 299): close[278] / close[47] - 1
        assert result.iloc[-1] == pytest.approx(closes[278] / closes[47] - 1)

    def test_rejects_non_multiindex(self):
        with pytest.raises(ValueError):
            MomentumFactor().compute(pd.DataFrame({"Close": [1, 2, 3]}))

    def test_positive_momentum_for_rising_prices(self):
        data = _single_ticker(np.arange(100, 130))  # strictly increasing
        result = MomentumFactor(lookback=20, skip=5).compute(data)
        assert result.iloc[-1] > 0


class TestSentimentFactor:
    def test_broadcasts_scores_across_dates(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["Ticker", "Date"])
        data = pd.DataFrame(index=idx)
        scores = {"AAPL": {"score": 0.8}, "MSFT": {"score": -0.2}}

        result = SentimentFactor().compute(data, sentiment_scores=scores)

        assert (result.xs("AAPL", level="Ticker") == 0.8).all()
        assert (result.xs("MSFT", level="Ticker") == -0.2).all()

    def test_missing_scores_are_nan(self):
        # NaN (not 0) so the z-score fills it with the cross-sectional average.
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["Ticker", "Date"])
        result = SentimentFactor().compute(pd.DataFrame(index=idx), sentiment_scores={"AAPL": 0.5})
        assert result.xs("MSFT", level="Ticker").isna().all()
        assert (result.xs("AAPL", level="Ticker") == 0.5).all()
        assert SentimentFactor().compute(pd.DataFrame(index=idx)).isna().all()


class TestVolumeFactor:
    def test_returns_series_over_full_index(self, price_data):
        data = price_data(["AAPL"], periods=40)
        result = VolumeFactor().compute(data)
        assert isinstance(result, pd.Series)
        assert result.index.equals(data.index)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_volume_spike_takes_sign_of_return(self):
        closes = np.full(25, 100.0)
        closes[-1] = 90.0  # sell-off on the last day
        volume = np.full(25, 1_000_000.0)
        volume[-1] = 5_000_000.0
        selloff = _single_ticker(closes, volume=volume)
        rally = _single_ticker(np.r_[closes[:-1], 110.0], volume=volume)
        assert VolumeFactor().compute(selloff).iloc[-1] < 0
        assert VolumeFactor().compute(rally).iloc[-1] > 0


class TestVolatilityFactor:
    def test_warm_up_is_nan(self, price_data):
        result = VolatilityFactor(window=20).compute(price_data(["AAPL"], periods=40))
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().all()

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

