from typing import List
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

        # Flatten MultiIndex columns if present (e.g., from Yahoo Finance)
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = ['_'.join(col).strip() for col in price_data.columns.values]

        price_data = self._add_technical_indicators(price_data)

        self._cache.set(cache_key, price_data, ttl=14400)
        return price_data

    def _add_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        tickers = price_data.index.get_level_values('Ticker').unique()

        for ticker in tickers:
            ticker_df = price_data.xs(ticker, level='Ticker').copy()
            ticker_df.columns = [col.lower() for col in ticker_df.columns]

            # Ensure close is a 1D Series
            close_series = ticker_df['close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            rsi = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
            macd = ta.trend.MACD(close=close_series)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            sma_20 = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()

            price_data.loc[ticker, 'RSI'] = rsi.values
            price_data.loc[ticker, 'MACD'] = macd_line.values
            price_data.loc[ticker, 'MACD_SIGNAL'] = macd_signal.values
            price_data.loc[ticker, 'SMA_20'] = sma_20.values
            price_data.loc[ticker, 'SMA_50'] = sma_50.values

        return price_data