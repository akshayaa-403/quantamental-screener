"""Tests for the Redis-backed cache client.

The default CacheClient short-circuits when Redis is disabled (the out-of-the-box
setting), so the ``fake_cache`` fixture is used to exercise the real get/set/
delete/clear logic against an in-memory FakeStrictRedis.
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


def test_delete_pattern_only_removes_matches(fake_cache):
    fake_cache.set("news:AAPL", ["h1"])
    fake_cache.set("news:MSFT", ["h2"])
    fake_cache.set("price:AAPL", [1, 2])

    removed = fake_cache.delete_pattern("news:*")

    assert removed == 2
    assert fake_cache.get("news:AAPL") is None
    assert fake_cache.get("price:AAPL") == [1, 2]


def test_clear_removes_everything(fake_cache):
    fake_cache.set("a", 1)
    fake_cache.set("b", 2)

    fake_cache.clear()

    assert fake_cache.get("a") is None
    assert fake_cache.get("b") is None
