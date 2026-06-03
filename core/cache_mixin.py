from .cache_client import CacheClient
import hashlib
import json

class CacheMixin:
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = CacheClient()
    
    def _cache_key(self, prefix: str, **kwargs) -> str:
        sorted_items = sorted(kwargs.items())
        args_str = json.dumps(sorted_items, sort_keys=True)
        hash_val = hashlib.md5(args_str.encode()).hexdigest()[:12]
        return f"{prefix}:{hash_val}"