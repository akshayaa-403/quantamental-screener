"""Smoke tests for the Streamlit dashboard (network-free).

The network-touching UI stages (universe, prices, news, headlines) and the SPY
download are patched with synthetic data; scoring and the backtest run for real.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from pipeline import BacktestEngine  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TICKERS = [f"T{i:02d}" for i in range(30)]
WITH_NEWS = TICKERS[:3]  # live sentiment comes back only for these


@pytest.fixture(scope="module", autouse=True)
def fresh_cache():
    # Once per module: every test feeds identical synthetic data, so the real
    # (slow) scoring and backtest stages can be reused from the cache.
    st.cache_data.clear()


@pytest.fixture
def app(price_data, monkeypatch):
    from ui import stages

    # ~2.8 years of business days ending today: covers a 1Y window plus factor warm-up.
    start = pd.Timestamp.today().normalize() - pd.offsets.BDay(699)
    prices = price_data(TICKERS, periods=700, start=start)

    def benchmark(self, start_date, end_date):
        idx = pd.bdate_range(start_date, end_date)
        return pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)

    monkeypatch.setattr(stages, "select_universe", lambda size: list(TICKERS))
    monkeypatch.setattr(stages, "download_prices", lambda tickers, start, end: prices)
    monkeypatch.setattr(stages, "live_sentiment", lambda tickers: {
        t: {"score": 0.4, "confidence": 0.7, "model": "ensemble"} for t in tickers if t in WITH_NEWS})
    monkeypatch.setattr(stages, "headlines", lambda ticker: [f"{ticker} beats $estimates", "Second headline"])
    monkeypatch.setattr(BacktestEngine, "_fetch_benchmark", benchmark)
    monkeypatch.setattr(BacktestEngine, "_fetch_risk_free",
                        lambda self, start_date, end_date: pd.Series(0.04, index=pd.bdate_range(start_date, end_date)))
    return AppTest.from_file(APP, default_timeout=120)


def run_screener(at):
    at.run()
    next(b for b in at.sidebar.button if b.label == "Run screener").click().run()
    assert not at.exception, at.exception
    return at


def texts(at):
    return " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.warning, *at.error, *at.info])


def test_empty_state_before_run(app):
    app.run()
    assert not app.exception
    assert app.button(key="demo_run").label.startswith("Run demo")
    assert "Not run yet" in texts(app)
    assert len(app.dataframe) == 0


def test_demo_run_uses_mock_sentiment(app):
    app.run()
    app.button(key="demo_run").click().run()
    assert not app.exception
    assert app.session_state["w_mock"] is True
    assert "Mock sentiment" in texts(app)
    assert "Settings changed" not in texts(app)


def test_every_page_renders_after_run(app):
    at = run_screener(app)
    # Overview (default page)
    assert [m.label for m in at.metric][:4] == ["Stocks screened", "Top pick", "Significance (p-value)",
                                                 "Strategy vs universe"]
    # Rankings
    at.button(key="ov_rankings").click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1
    assert "Price-only rank" in at.dataframe[0].value.columns
    # Backtest (a plain rerun returns AppTest to the default page first)
    at.run().button(key="ov_backtest").click().run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}
    assert {"Strategy return", "vs equal-weight universe", "vs SPY", "Sharpe ratio", "Max drawdown",
            "Turnover", "Costs paid", "Universe EW", "p-value"} <= labels
    assert "Holdout" in texts(at)
    assert "survivorship and size bias" in texts(at).lower()
    # Stock detail, opened from a top-pick card
    top = at.run().metric[1].value
    at.button(key=f"ov_open_{top}").click().run()
    assert not at.exception, at.exception
    assert at.session_state["selected_ticker"] == top
    assert at.selectbox(key="detail_ticker").value == top


def test_unfetched_sentiment_is_blank_not_zero(app):
    at = run_screener(app)
    at.session_state["rank_top_n"] = "All"
    at.button(key="ov_rankings").click().run()
    assert not at.exception, at.exception
    table = at.dataframe[0].value.set_index("Ticker")
    assert len(table) == len(TICKERS)
    sentiment = table["Sentiment"]
    assert sentiment[WITH_NEWS].notna().all()
    without = [t for t in TICKERS if t not in WITH_NEWS]
    assert sentiment[without].isna().all()  # blank, not 0.000


def test_weight_change_marks_results_stale(app):
    at = run_screener(app)
    assert "Settings changed" not in texts(at)
    at.sidebar.slider(key="w_momentum").set_value(0.9).run()
    assert not at.exception
    assert "Settings changed" in texts(at)


def test_in_app_navigation_keeps_settings(app):
    # st.switch_page drops widget state; the app must restore the sidebar settings.
    app.run()
    app.button(key="demo_run").click().run()
    top = app.metric[1].value
    app.button(key=f"ov_open_{top}").click().run()
    assert not app.exception, app.exception
    assert app.session_state["w_mock"] is True
    assert "Settings changed" not in texts(app)


def test_empty_download_is_reported_and_not_cached(monkeypatch):
    from pipeline import DataCollector
    from ui import stages

    calls = []

    def collect(self, tickers, start, end):
        calls.append(1)
        return pd.DataFrame()

    monkeypatch.setattr(DataCollector, "collect", collect)
    for _ in range(2):
        with pytest.raises(stages.ScreenerError, match="no price data"):
            stages.download_prices(("ZZZ",), "2025-01-01", "2025-06-01")
    assert len(calls) == 2  # the empty result was retried, not replayed from the cache


def test_presets_compare_by_label_not_by_date():
    from ui.state import same_settings

    base = {"universe_size": 50, "start_date": "2025-11-08", "end_date": "2026-11-08",
            "weights": (("momentum", 1.0),), "use_mock": True, "period": "1Y"}
    next_day = {**base, "start_date": "2025-11-09", "end_date": "2026-11-09"}
    assert same_settings(base, next_day)  # "1Y" rolled past midnight: not a settings change
    assert not same_settings(base, {**base, "period": "2Y"})
    custom = {**base, "period": "Custom"}
    assert not same_settings(custom, {**custom, "end_date": "2026-10-01"})
    assert not same_settings(base, None)
