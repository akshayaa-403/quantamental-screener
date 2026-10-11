from typing import List
import pandas as pd
from core.cache_mixin import CacheMixin
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class DataCollector(CacheMixin):
    def __init__(self, data_source):
        CacheMixin.__init__(self)
        self.data_source = data_source

    def collect(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = self._cache_key("collected_data", tickers=sorted(tickers), start=start_date, end=end_date)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Returning cached collected data")
            return cached

        price_data = self.data_source.get_price_data(tickers, start_date, end_date)
        self._cache.set(cache_key, price_data, ttl=get_settings().redis.ttl_price)
        return price_data
