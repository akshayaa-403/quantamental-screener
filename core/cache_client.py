import redis
import pickle
from typing import Any, Optional
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

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
        # Values are unpickled, so only point this at a Redis nobody else can write to.
        return None if data is None else pickle.loads(data)

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self.enabled:
            return False
        self.client.setex(key, ttl or self.default_ttl, pickle.dumps(value))
        return True

    def clear(self):
        if self.enabled:
            self.client.flushdb()
