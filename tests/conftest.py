import pytest
from fakeredis import FakeStrictRedis
from core.cache_client import CacheClient

@pytest.fixture
def mock_redis():
    client = FakeStrictRedis()
    CacheClient.client = client
    return client