from typing import List, Dict
import pandas as pd
from core.data_source import DataSource
from core.cache_mixin import CacheMixin
import ta
import logging

logger = logging.getLogger(__name__)

class DataCollector(CacheMixin):
    def __init__(self, data_source: DataSource):
        CacheMixin.__init__(self)
        self.data_source = data_source

    def collect(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = self._cache_key("collected_data", tickers=sorted(tickers), start=start_date, end=end_date)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Returning cached collected data")
            return cached

        price_data = self.data_source.get_price_data(tickers, start_date, end_date)
        price_data = self._add_technical_indicators(price_data)

        self._cache.set(cache_key, price_data, ttl=14400)
        return price_data

    def _add_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators (RSI, MACD, etc.) to the price data.
        Works regardless of column name case.
        """
        for ticker in price_data.columns.levels[0]:
            ticker_df = price_data[ticker].copy()

            ticker_df.columns = [col.lower() for col in ticker_df.columns]

            ticker_df['rsi'] = ta.momentum.RSIIndicator(close=ticker_df['close'], window=14).rsi()

            macd = ta.trend.MACD(close=ticker_df['close'])
            ticker_df['macd'] = macd.macd()
            ticker_df['macd_signal'] = macd.macd_signal()

            ticker_df['sma_20'] = ta.trend.SMAIndicator(close=ticker_df['close'], window=20).sma_indicator()
            ticker_df['sma_50'] = ta.trend.SMAIndicator(close=ticker_df['close'], window=50).sma_indicator()

            for col in ticker_df.columns:
                price_data[(ticker, col.upper())] = ticker_df[col]

        return price_data