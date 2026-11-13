"""Cached pipeline stages for the dashboard.

Each stage is a thin wrapper over the unchanged pipeline classes, split up so
the UI can report progress per stage. Cache keys come only from hashable
arguments (strings, numbers, tuples, DataFrames, dicts), never from the
heavy components, which come from ``load_components``.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import WARMUP_DAYS, get_settings
from data_sources.yahoo_finance import YahooFinanceSource
from factors import MomentumFactor, SentimentFactor, VolatilityFactor, VolumeFactor
from pipeline import BacktestEngine, DataCollector, ScoringEngine, UniverseSelector
from sentiment.ensemble_sentiment import EnsembleSentiment

TTL = 300  # seconds; results are cached for 5 minutes
NEWS_WORKERS = 8  # parallel news requests; every name in the universe gets news

EMPTY_PRICES = ("Yahoo Finance returned no price data. It may be rate-limiting this server or be briefly "
                "unavailable. Wait a minute, then run again.")


class ScreenerError(Exception):
    """A pipeline outcome with a known, user-readable explanation.

    Raised inside the cached stages, because Streamlit never caches an exception:
    an empty download is retried on the next run instead of replayed for 5 minutes.
    """


class NoHeadlines(Exception):
    """No live headlines came back for any ticker (raised so the empty result isn't cached)."""


@st.cache_resource(ttl=3600)
def load_components():
    """Load heavy components once and cache them."""
    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor()]
    return data_source, sentiment_model, factors


@st.cache_data(ttl=TTL, show_spinner=False)
def select_universe(universe_size: int) -> list:
    tickers = UniverseSelector({"size": universe_size}).select()
    if not tickers:
        raise ScreenerError("No tickers passed the universe filters.")
    return tickers


@st.cache_data(ttl=TTL, show_spinner=False)
def download_prices(tickers: tuple, start_date: str, end_date: str) -> pd.DataFrame:
    # WARMUP_DAYS of extra history before the chosen start so 12-minus-1
    # momentum can score from the first day of the backtest.
    data_start = (pd.Timestamp(start_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    data_source, _, _ = load_components()
    prices = DataCollector(data_source).collect(list(tickers), data_start, end_date)
    if prices.empty:
        raise ScreenerError(EMPTY_PRICES)
    return prices


@st.cache_data(ttl=TTL, show_spinner=False)
def mock_sentiment(tickers: tuple) -> dict:
    """Seeded mock sentiment: fast, offline, reproducible."""
    rng = np.random.default_rng(42)
    return {t: {"score": float(rng.uniform(-0.8, 0.8))} for t in tickers}


@st.cache_data(ttl=3600, show_spinner=False)
def company_names() -> dict:
    """Ticker -> company name from the S&P 500 list, for precise news searches."""
    selector = UniverseSelector()
    selector._fetch_sp500_csv()
    return selector.company_names


def fetch_news(data_source, tickers, names: dict) -> dict:
    """Ticker -> recent headlines, fetched in parallel; failures count as no news."""
    cfg = get_settings().sentiment

    def one(ticker):
        try:
            return ticker, data_source.get_news_headlines(
                ticker, lookback_days=cfg.lookback_days, company_name=names.get(ticker),
                max_headlines=cfg.max_headlines)
        except Exception:  # noqa: BLE001 - one name's news failing shouldn't stop the run
            return ticker, []

    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as pool:
        return {t: h for t, h in pool.map(one, tickers) if h}


@st.cache_data(ttl=TTL, show_spinner=False)
def live_sentiment(tickers: tuple) -> dict:
    """News sentiment for every ticker; names without headlines are left out (scored neutral)."""
    data_source, sentiment_model, _ = load_components()
    news_by_ticker = fetch_news(data_source, tickers, company_names())
    if not news_by_ticker:
        raise NoHeadlines()
    return sentiment_model.batch_analyze(news_by_ticker)


@st.cache_data(ttl=TTL, show_spinner=False)
def score(price_data: pd.DataFrame, sentiment_scores: dict, weights: tuple) -> pd.DataFrame:
    # The engine keeps its own copy of the weights, so the cached factor
    # instances are never mutated between runs.
    _, _, factors = load_components()
    return ScoringEngine(factors, weights=dict(weights)).compute(price_data, sentiment_scores)


@st.cache_data(ttl=TTL, show_spinner=False)
def backtest(price_data: pd.DataFrame, weights: tuple, start_date: str) -> dict:
    # Backtest without sentiment: today's news on past dates is lookahead.
    _, _, factors = load_components()
    scorer = ScoringEngine(factors, weights=dict(weights))
    return BacktestEngine().run(scorer, price_data, sentiment_scores=None, start_date=start_date)


@st.cache_data(ttl=600, show_spinner=False)
def headlines(ticker: str) -> list:
    data_source, _, _ = load_components()
    cfg = get_settings().sentiment
    return data_source.get_news_headlines(ticker, lookback_days=cfg.lookback_days,
                                          company_name=company_names().get(ticker),
                                          max_headlines=cfg.max_headlines)
