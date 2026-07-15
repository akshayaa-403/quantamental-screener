"""Integration tests for the pipeline components.

These tests are network-free: universe fetching and benchmark downloads are
stubbed, and the data source is mocked.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from factors import MomentumFactor, VolatilityFactor, VolumeFactor
from pipeline import BacktestEngine, DataCollector, ScoringEngine, UniverseSelector


class TestUniverseSelector:
    def test_respects_universe_size(self):
        selector = UniverseSelector({"size": 3})
        # Stub the (network) primary fetch with a known list.
        selector._fetch_sp500_yfinance = lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        assert selector.select() == ["AAPL", "MSFT", "GOOGL"]

    def test_falls_back_to_hardcoded_when_sources_empty(self):
        selector = UniverseSelector({"size": 4})
        selector._fetch_sp500_yfinance = lambda: []
        selector._fetch_sp500_csv_fallback = lambda: []
        tickers = selector.select()
        assert len(tickers) == 4
        assert all(isinstance(t, str) and t for t in tickers)


class TestDataCollector:
    def test_adds_technical_indicators(self, price_data):
        source = Mock()
        source.get_price_data.return_value = price_data(["AAPL"], periods=60)
        collector = DataCollector(source)

        result = collector.collect(["AAPL"], "2024-01-01", "2024-03-01")

        for col in ["RSI", "MACD", "MACD_SIGNAL", "SMA_20", "SMA_50"]:
            assert col in result.columns

    def test_preserves_all_tickers(self, price_data):
        source = Mock()
        source.get_price_data.return_value = price_data(["AAPL", "MSFT"], periods=40)
        collector = DataCollector(source)

        result = collector.collect(["AAPL", "MSFT"], "2024-01-01", "2024-03-01")

        assert set(result.index.get_level_values("Ticker").unique()) == {"AAPL", "MSFT"}


class TestScoringEngine:
    def test_produces_composite_and_per_date_rank(self, price_data):
        data = price_data(["AAPL", "MSFT", "GOOGL"], periods=40)
        engine = ScoringEngine(
            [MomentumFactor(), VolumeFactor(), VolatilityFactor()],
            weights={"momentum": 0.5, "volume": 0.3, "volatility": 0.2},
        )

        result = engine.compute(data, cross_sectional_normalize=True)

        assert "composite" in result.columns and "rank" in result.columns
        last_date = result.index.get_level_values("Date").max()
        ranks = result.xs(last_date, level="Date")["rank"]
        assert set(ranks) <= {1.0, 2.0, 3.0}

    def test_applies_weights_to_composite(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["Ticker", "Date"])
        data = pd.DataFrame(index=idx)

        momentum, volume = MomentumFactor(), VolumeFactor()
        # AAPL > MSFT on momentum, MSFT > AAPL on volume. After per-date z-score
        # standardization a two-name cross-section maps to {-1, +1}.
        momentum.compute = Mock(
            return_value=pd.Series(
                [0.5, 0.5, -0.3, -0.3], index=idx, name="momentum"
            )
        )
        volume.compute = Mock(
            return_value=pd.Series(
                [0.1, 0.1, 0.9, 0.9], index=idx, name="volume"
            )
        )

        engine = ScoringEngine([momentum, volume], weights={"momentum": 0.6, "volume": 0.4})
        result = engine.compute(data, cross_sectional_normalize=True)

        day0 = result.xs(dates[0], level="Date")
        # AAPL: 0.6*(+1) + 0.4*(-1) = 0.2 ; MSFT: 0.6*(-1) + 0.4*(+1) = -0.2
        assert day0.loc["AAPL", "composite"] == pytest.approx(0.2)
        assert day0.loc["MSFT", "composite"] == pytest.approx(-0.2)
        assert day0.loc["AAPL", "rank"] == 1.0


class TestBacktestEngine:
    def test_empty_price_data_returns_empty(self):
        engine = BacktestEngine()
        scorer = ScoringEngine([MomentumFactor()])
        assert engine.run(scorer, pd.DataFrame()) == {}

    def test_run_produces_metrics(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        dates = data.index.get_level_values("Date").unique().sort_values()

        engine = BacktestEngine({"top_n": 2, "rebalance_freq": 5})
        # Avoid the network: synthetic benchmark aligned to the price dates.
        monkeypatch.setattr(
            engine,
            "_fetch_benchmark",
            lambda start, end: pd.Series(np.linspace(100, 110, len(dates)), index=dates),
        )
        scorer = ScoringEngine(
            [MomentumFactor(), VolumeFactor(), VolatilityFactor()],
            weights={"momentum": 0.5, "volume": 0.2, "volatility": 0.3},
        )

        result = engine.run(scorer, data, sentiment_scores=None)

        for key in [
            "strategy_return",
            "benchmark_return",
            "excess_return",
            "sharpe_ratio",
            "max_drawdown",
            "volatility",
            "hit_rate",
            "final_value",
            "initial_capital",
        ]:
            assert key in result
        assert isinstance(result["portfolio_series"], pd.Series)
        assert len(result["portfolio_series"]) == len(dates)
        # The equity curve starts at the reported initial capital, so a chart
        # normalized off it lines up with the reported strategy return.
        assert result["initial_capital"] == 1_000_000
        assert result["portfolio_series"].iloc[0] == pytest.approx(1_000_000, rel=1e-6)

    def test_risk_free_rate_lowers_sharpe(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        dates = data.index.get_level_values("Date").unique().sort_values()
        scorer = ScoringEngine([MomentumFactor(), VolumeFactor(), VolatilityFactor()])
        bench = lambda start, end: pd.Series(np.linspace(100, 110, len(dates)), index=dates)

        no_rf = BacktestEngine({"top_n": 2, "rebalance_freq": 5, "risk_free_rate": 0.0})
        monkeypatch.setattr(no_rf, "_fetch_benchmark", bench)
        with_rf = BacktestEngine({"top_n": 2, "rebalance_freq": 5, "risk_free_rate": 0.05})
        monkeypatch.setattr(with_rf, "_fetch_benchmark", bench)

        s0 = no_rf.run(scorer, data, sentiment_scores=None)["sharpe_ratio"]
        s1 = with_rf.run(scorer, data, sentiment_scores=None)["sharpe_ratio"]
        assert s1 < s0
