"""Tests for the Redis-backed cache client.

The default CacheClient short-circuits when Redis is disabled (the out-of-the-box
setting), so the ``fake_cache`` fixture is used to exercise the real get/set/
clear logic against an in-memory FakeStrictRedis.
"""

import pandas as pd

from core.cache_client import CacheClient


def test_disabled_cache_is_noop():
    """With Redis disabled (default), set is a no-op and get returns None."""
    cache = CacheClient()
    assert cache.enabled is False
    assert cache.set("k", "v") is False
    assert cache.get("k") is None


def test_set_get_roundtrip(fake_cache):
    fake_cache.set("greeting", "hello", ttl=10)
    assert fake_cache.get("greeting") == "hello"


def test_get_missing_key_returns_none(fake_cache):
    assert fake_cache.get("does-not-exist") is None


def test_roundtrip_dataframe(fake_cache):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    fake_cache.set("frame", df)
    pd.testing.assert_frame_equal(fake_cache.get("frame"), df)


def test_roundtrip_price_panel(fake_cache, price_data):
    # The real cached payload: a (Ticker, Date) MultiIndex OHLCV frame.
    df = price_data(["AAPL", "MSFT"], periods=5)
    fake_cache.set("prices", df)
    pd.testing.assert_frame_equal(fake_cache.get("prices"), df, check_freq=False)


def test_roundtrip_headlines(fake_cache):
    fake_cache.set("news", ["Apple beats estimates", "Second"])
    assert fake_cache.get("news") == ["Apple beats estimates", "Second"]


def test_never_unpickles(fake_cache):
    # A pickle planted in Redis must be ignored, not executed.
    import pickle
    fake_cache.client.set("evil", pickle.dumps({"a": 1}))
    assert fake_cache.get("evil") is None


def test_unserializable_value_is_not_cached(fake_cache):
    assert fake_cache.set("obj", object()) is False
    assert fake_cache.get("obj") is None


def test_set_uses_ttl(fake_cache):
    fake_cache.set("short", "v", ttl=30)
    assert 0 < fake_cache.client.ttl("short") <= 30


def test_clear_removes_everything(fake_cache):
    fake_cache.set("a", 1)
    fake_cache.set("b", 2)

    fake_cache.clear()

    assert fake_cache.get("a") is None
    assert fake_cache.get("b") is None
