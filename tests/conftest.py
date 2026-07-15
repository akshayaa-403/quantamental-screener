"""Shared pytest fixtures.

Provides deterministic, network-free building blocks for the test suite:
- ``fake_cache``: a ``CacheClient`` backed by in-memory ``FakeStrictRedis`` with
  caching enabled, so the real get/set/delete logic can be exercised (the
  default client short-circuits when Redis is disabled).
- ``price_data``: a factory that builds synthetic OHLCV data with the
  ``(Ticker, Date)`` MultiIndex that the factors and pipeline expect.
"""

import numpy as np
import pandas as pd
import pytest
from fakeredis import FakeStrictRedis

from core.cache_client import CacheClient


@pytest.fixture
def fake_cache():
    """A CacheClient wired to an in-memory FakeStrictRedis, with caching on."""
    cache = CacheClient()
    cache.enabled = True
    cache.client = FakeStrictRedis()
    return cache


@pytest.fixture
def price_data():
    """Factory building a synthetic OHLCV DataFrame indexed by (Ticker, Date)."""

    def _make(tickers, periods=60, seed=0, start="2024-01-01"):
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, periods=periods, freq="B")
        frames = []
        for i, ticker in enumerate(tickers):
            # Deterministic per-ticker drift plus noise -> distinct series.
            steps = rng.normal(0.001 * (i + 1), 0.01, size=periods)
            close = 100 * np.cumprod(1 + steps)
            df = pd.DataFrame(
                {
                    "Open": close * 0.99,
                    "High": close * 1.01,
                    "Low": close * 0.98,
                    "Close": close,
                    "Volume": rng.integers(1_000_000, 5_000_000, size=periods),
                }
            )
            df["Ticker"] = ticker
            df["Date"] = dates
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        return combined.set_index(["Ticker", "Date"]).sort_index()

    return _make
