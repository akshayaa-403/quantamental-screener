import io
import json
import logging
from typing import Any, Optional

import pandas as pd
import redis

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Values are stored as data, never as pickles: unpickling runs arbitrary code,
# so anyone able to write to the Redis instance could otherwise run code here.
# DataFrames go through Parquet (keeps the (Ticker, Date) MultiIndex and
# dtypes); everything else must be JSON.
_FRAME, _JSON = b"P", b"J"


def _encode(value: Any) -> bytes:
    if isinstance(value, pd.DataFrame):
        buffer = io.BytesIO()
        value.to_parquet(buffer)
        return _FRAME + buffer.getvalue()
    return _JSON + json.dumps(value).encode("utf-8")


def _decode(data: bytes) -> Any:
    tag, body = data[:1], data[1:]
    if tag == _FRAME:
        return pd.read_parquet(io.BytesIO(body))
    if tag == _JSON:
        return json.loads(body.decode("utf-8"))
    raise ValueError("Unrecognized cache entry")


class CacheClient:
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.redis.enabled
        self.default_ttl = settings.redis.ttl_default
        self.client = None
        if self.enabled:
            try:
                self.client = redis.Redis.from_url(settings.redis.url, decode_responses=False)
                self.client.ping()
                logger.info(f"Connected to Redis at {settings.redis.url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        data = self.client.get(key)
        if data is None:
            return None
        try:
            return _decode(data)
        except Exception as e:  # noqa: BLE001 - a bad entry (e.g. an old pickle) is a cache miss
            logger.warning(f"Ignoring unreadable cache entry {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self.enabled:
            return False
        try:
            payload = _encode(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Not caching {key}: {e}")
            return False
        self.client.set(key, payload, ex=ttl or self.default_ttl)
        return True

    def clear(self):
        if self.enabled:
            self.client.flushdb()
