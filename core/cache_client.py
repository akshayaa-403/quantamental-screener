import redis
import pickle
import json
from typing import Any, Optional
from fakeredis import FakeStrictRedis
from config import settings
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class CacheClient:
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.redis.enabled
        self.default_ttl = settings.redis.ttl_default
        if self.enabled:
            try:
                self.client = redis.Redis.from_url(settings.redis.url, decode_responses=False)
                self.client.ping()
                logger.info(f"Connected to Redis at {settings.redis.url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to fake Redis.")
                self.client = FakeStrictRedis()
                self.enabled = False  
        else:
            self.client = FakeStrictRedis()
    
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        data = self.client.get(key)
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except (pickle.PickleError, TypeError):
            # fallback to JSON if stored as string
            return json.loads(data.decode('utf-8'))
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self.enabled:
            return False
        ttl = ttl or self.default_ttl
        try:
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
        except (pickle.PickleError, TypeError):
            # fallback to JSON
            serialized = json.dumps(value).encode('utf-8')
            self.client.setex(key, ttl, serialized)
        return True
    
    def delete_pattern(self, pattern: str) -> int:
        if not self.enabled:
            return 0
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def clear(self):
        if self.enabled:
            self.client.flushdb()