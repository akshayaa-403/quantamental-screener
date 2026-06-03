from core.cache_client import CacheClient

if __name__ == "__main__":
    cache = CacheClient()
    cache.clear()
    print("Redis cache cleared.")