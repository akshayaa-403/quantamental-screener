import pytest
from core.cache_client import CacheClient

def test_cache_set_get():
    cache = CacheClient()
    cache.set("test_key", "test_value", ttl=10)
    assert cache.get("test_key") == "test_value"