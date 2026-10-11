"""Integration tests for the pipeline components.

These tests are network-free: universe fetching and benchmark downloads are
stubbed, and the data source is mocked.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from factors import MomentumFactor, SentimentFactor, VolatilityFactor, VolumeFactor
from pipeline import BacktestEngine, DataCollector, ScoringEngine, UniverseSelector


class TestUniverseSelector:
    def test_respects_universe_size(self):
        selector = UniverseSelector({"size": 3, "min_market_cap": 0, "min_volume": 0})
        # Stub the (network) constituent fetch with a known list.
        selector._fetch_sp500_csv = lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        assert selector.select() == ["AAPL", "MSFT", "GOOGL"]

    def test_filters_and_orders_by_market_cap(self):
        selector = UniverseSelector({"size": 2, "min_market_cap": 5e9, "min_volume": 1e6})
        selector._fetch_sp500_csv = lambda: ["SMALL", "THIN", "MID", "BIG", "UNKNOWN"]
        selector._fetch_fundamentals = lambda tickers: {
            "SMALL": (1e9, 5e6),      # fails market cap
            "THIN": (50e9, 1e5),      # fails volume
            "MID": (20e9, 2e6),
            "BIG": (900e9, 3e7),
            "UNKNOWN": (None, None),  # no data -> excluded
        }
        assert selector.select() == ["BIG", "MID"]

    def test_unfiltered_when_fundamentals_unavailable(self):
        selector = UniverseSelector({"size": 2})
        selector._fetch_sp500_csv = lambda: ["AAPL", "MSFT", "GOOGL"]
        selector._fetch_fundamentals = lambda tickers: {}
        assert selector.select() == ["AAPL", "MSFT"]

    def test_share_classes_use_yahoo_spelling(self):
        selector = UniverseSelector({"size": 5, "min_market_cap": 0, "min_volume": 0})
        selector._fetch_sp500_csv = lambda: ["AAPL", "BRK.B", "BF.B"]
        assert selector.select() == ["AAPL", "BRK-B", "BF-B"]

    def test_falls_back_to_hardcoded_when_sources_empty(self):
        selector = UniverseSelector({"size": 4, "min_market_cap": 0, "min_volume": 0})
        selector._fetch_sp500_csv = lambda: []
        tickers = selector.select()
        assert len(tickers) == 4
        assert all(isinstance(t, str) and t for t in tickers)


class TestDataCollector:
    def test_returns_source_data(self, price_data):
        source = Mock()
        source.get_price_data.return_value = price_data(["AAPL", "MSFT"], periods=40)
        result = DataCollector(source).collect(["AAPL", "MSFT"], "2024-01-01", "2024-03-01")
        assert set(result.index.get_level_values("Ticker").unique()) == {"AAPL", "MSFT"}


def _short_factors():
    """Factors with short windows so synthetic histories warm up quickly."""
    return [MomentumFactor(lookback=20, skip=5), VolumeFactor(), VolatilityFactor()]


class TestScoringEngine:
    def test_produces_composite_and_per_date_rank(self, price_data):
        data = price_data(["AAPL", "MSFT", "GOOGL"], periods=40)
        engine = ScoringEngine(
            _short_factors(),
            weights={"momentum": 0.5, "volume": 0.3, "volatility": 0.2},
        )

        result = engine.compute(data)

        assert "composite" in result.columns and "rank" in result.columns
        last_date = result.index.get_level_values("Date").max()
        ranks = result.xs(last_date, level="Date")["rank"]
        assert set(ranks) <= {1.0, 2.0, 3.0}

    def test_warm_up_rows_have_no_composite(self, price_data):
        data = price_data(["AAPL", "MSFT", "GOOGL"], periods=40)
        result = ScoringEngine(_short_factors()).compute(data)
        first = result.xs(data.index.get_level_values("Date")[0], level="Date")
        assert first["composite"].isna().all()
        assert first["rank"].isna().all()
        last = result.xs(data.index.get_level_values("Date").max(), level="Date")
        assert last["composite"].notna().all()

    def test_does_not_mutate_factor_weights(self):
        momentum = MomentumFactor()
        ScoringEngine([momentum], weights={"momentum": 0.7})
        assert momentum.weight == 1.0

    def test_missing_sentiment_is_neutral(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        idx = pd.MultiIndex.from_product([["A", "B", "C"], dates], names=["Ticker", "Date"])
        engine = ScoringEngine([SentimentFactor()])
        # Only A and B have news; C has none and must land exactly on average,
        # not below it as a raw 0 would when the covered names average +0.2.
        result = engine.compute(pd.DataFrame(index=idx), {"A": 0.4, "B": 0.0})
        day0 = result.xs(dates[0], level="Date")
        assert day0.loc["C", "sentiment"] == pytest.approx(0.0)
        assert day0.loc["A", "sentiment"] > 0 > day0.loc["B", "sentiment"]

    def test_future_prices_do_not_move_past_scores(self, price_data):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=3)
        cutoff = data.index.get_level_values("Date").unique()[40]
        shocked = data.copy()
        future = shocked.index.get_level_values("Date") > cutoff
        shocked.loc[future, "Close"] *= 3.0
        shocked.loc[future, "Volume"] *= 10

        engine = ScoringEngine(_short_factors())
        before = engine.compute(data)
        after = engine.compute(shocked)

        past = before.index.get_level_values("Date") <= cutoff
        pd.testing.assert_frame_equal(before[past], after[past])

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
        result = engine.compute(data)

        day0 = result.xs(dates[0], level="Date")
        # AAPL: 0.6*(+1) + 0.4*(-1) = 0.2 ; MSFT: 0.6*(-1) + 0.4*(+1) = -0.2
        assert day0.loc["AAPL", "composite"] == pytest.approx(0.2)
        assert day0.loc["MSFT", "composite"] == pytest.approx(-0.2)
        assert day0.loc["AAPL", "rank"] == 1.0


def _engine(data, monkeypatch, **config):
    config = {"top_n": 2, "rebalance_freq": 5, "transaction_cost_bps": 0.0,
              "min_volume": 0, "n_bootstrap": 200, **config}
    engine = BacktestEngine(config)
    # Avoid the network: synthetic benchmark aligned to the price dates.
    dates = data.index.get_level_values("Date").unique().sort_values()
    monkeypatch.setattr(
        engine,
        "_fetch_benchmark",
        lambda start, end: pd.Series(np.linspace(100, 110, len(dates)), index=dates),
    )
    return engine


class TestBacktestEngine:
    def test_empty_price_data_returns_empty(self):
        engine = BacktestEngine()
        scorer = ScoringEngine([MomentumFactor()])
        assert engine.run(scorer, pd.DataFrame()) == {}

    def test_too_short_for_warm_up_returns_empty(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL"], periods=60)
        # Default 12-1 momentum needs 252 days of history.
        assert _engine(data, monkeypatch).run(ScoringEngine([MomentumFactor()]), data) == {}

    def test_run_produces_metrics(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        scorer = ScoringEngine(
            _short_factors(), weights={"momentum": 0.5, "volume": 0.2, "volatility": 0.3}
        )

        result = _engine(data, monkeypatch).run(scorer, data, sentiment_scores=None)

        for key in [
            "strategy_return", "benchmark_return", "excess_return", "sharpe_ratio",
            "max_drawdown", "volatility", "hit_rate", "p_value", "n_rebalances",
            "turnover", "total_costs", "final_value", "initial_capital",
        ]:
            assert key in result
        assert 0 < result["p_value"] <= 1
        series = result["portfolio_series"]
        assert isinstance(series, pd.Series)
        # Trading starts once every factor is warmed up (20-day windows) and
        # one more day for the signal lag, not on the first day of data.
        assert series.index[0] == data.index.get_level_values("Date").unique()[21]
        # The equity curve starts at the reported initial capital, so a chart
        # normalized off it lines up with the reported strategy return.
        assert result["initial_capital"] == 1_000_000
        assert series.iloc[0] == pytest.approx(1_000_000, rel=1e-6)
        assert result["benchmark_series"].iloc[0] == pytest.approx(1_000_000)

    def test_trades_on_previous_days_signal(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], periods=80, seed=5)
        scorer = ScoringEngine(_short_factors())
        base = _engine(data, monkeypatch).run(scorer, data)["portfolio_series"]

        # Shock the volume factor on a rebalance day. The picks made that day
        # must not see it, so the curve is unchanged until the next rebalance.
        rebalance = base.index[10]
        shocked = data.copy()
        on_day = shocked.index.get_level_values("Date") == rebalance
        shocked.loc[on_day, "Volume"] *= np.array([1, 1, 1, 1, 50])
        after = _engine(shocked, monkeypatch).run(scorer, shocked)["portfolio_series"]

        until = base.index[14]
        pd.testing.assert_series_equal(base.loc[:until], after.loc[:until])

    def test_future_prices_do_not_move_past_equity(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=80, seed=9)
        scorer = ScoringEngine(_short_factors())
        base = _engine(data, monkeypatch).run(scorer, data)["portfolio_series"]

        cutoff = base.index[30]
        shocked = data.copy()
        shocked.loc[shocked.index.get_level_values("Date") > cutoff, "Close"] *= 0.5
        after = _engine(shocked, monkeypatch).run(scorer, shocked)["portfolio_series"]

        pd.testing.assert_series_equal(base.loc[:cutoff], after.loc[:cutoff])

    def test_ignores_sentiment(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        scorer = ScoringEngine(_short_factors() + [SentimentFactor()], weights={"sentiment": 0.9})
        without = _engine(data, monkeypatch).run(scorer, data)
        with_news = _engine(data, monkeypatch).run(scorer, data, {"AAPL": 1.0, "MSFT": -1.0})
        pd.testing.assert_series_equal(without["portfolio_series"], with_news["portfolio_series"])
        # ...and leaves the live ranking's weight alone.
        assert scorer.weights["sentiment"] == 0.9

    def test_transaction_costs_lower_return(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        scorer = ScoringEngine(_short_factors())
        free = _engine(data, monkeypatch).run(scorer, data)
        costly = _engine(data, monkeypatch, transaction_cost_bps=50).run(scorer, data)
        assert costly["final_value"] < free["final_value"]
        assert costly["total_costs"] > 0 and free["total_costs"] == 0

    def test_only_changed_names_are_traded(self, monkeypatch):
        # Steady, distinct growth rates: the momentum ranking never changes, so
        # after the first purchase there is nothing to trade.
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        frames = [
            pd.DataFrame({
                "Ticker": ticker, "Date": dates,
                "Close": 100 * (1 + growth) ** np.arange(60), "Volume": 2_000_000,
            })
            for ticker, growth in [("A", 0.004), ("B", 0.003), ("C", 0.002), ("D", 0.001)]
        ]
        data = pd.concat(frames).set_index(["Ticker", "Date"]).sort_index()
        scorer = ScoringEngine([MomentumFactor(lookback=20, skip=5)])
        result = _engine(data, monkeypatch).run(scorer, data)
        assert result["n_rebalances"] > 1
        assert result["turnover"] == pytest.approx(1.0)

    def test_start_date_limits_trading_window(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=80, seed=7)
        start = data.index.get_level_values("Date").unique()[50]
        result = _engine(data, monkeypatch).run(
            ScoringEngine(_short_factors()), data, start_date=str(start.date())
        )
        assert result["portfolio_series"].index[0] == start

    def test_risk_free_rate_lowers_sharpe(self, price_data, monkeypatch):
        data = price_data(["AAPL", "MSFT", "GOOGL", "AMZN"], periods=60, seed=7)
        scorer = ScoringEngine(_short_factors())
        s0 = _engine(data, monkeypatch, risk_free_rate=0.0).run(scorer, data)["sharpe_ratio"]
        s1 = _engine(data, monkeypatch, risk_free_rate=0.05).run(scorer, data)["sharpe_ratio"]
        assert s1 < s0

    def test_benchmark_includes_last_day_and_adjusts(self, monkeypatch):
        # yfinance's end is exclusive: asking for end_date itself drops the last session.
        seen = {}
        def fake_download(ticker, start, end, auto_adjust, progress):
            seen.update(end=end, auto_adjust=auto_adjust)
            return pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-05"]))
        monkeypatch.setattr("pipeline.backtest_engine.yf.download", fake_download)
        BacktestEngine()._fetch_benchmark("2024-01-01", "2024-01-05")
        assert seen == {"end": "2024-01-06", "auto_adjust": True}
