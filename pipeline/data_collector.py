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
        
        # Get raw price data
        price_data = self.data_source.get_price_data(tickers, start_date, end_date)
        # Add technical indicators
        price_data = self._add_technical_indicators(price_data)
        
        self._cache.set(cache_key, price_data, ttl=14400)
        return price_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # df has MultiIndex (Ticker, Date)
        result = []
        for ticker in df.index.get_level_values('Ticker').unique():
            ticker_df = df.xs(ticker, level='Ticker').copy()
            ticker_df = ticker_df.sort_index()
            # RSI
            ticker_df['RSI'] = ta.momentum.RSIIndicator(close=ticker_df['Close'], window=14).rsi()
            # MACD
            macd = ta.trend.MACD(close=ticker_df['Close'])
            ticker_df['MACD'] = macd.macd()
            ticker_df['MACD_signal'] = macd.macd_signal()
            ticker_df['MACD_diff'] = macd.macd_diff()
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=ticker_df['Close'])
            ticker_df['BB_high'] = bb.bollinger_hband()
            ticker_df['BB_low'] = bb.bollinger_lband()
            ticker_df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / ticker_df['Close']
            # Volume moving average
            ticker_df['Volume_MA20'] = ticker_df['Volume'].rolling(20).mean()
            ticker_df['Volume_Ratio'] = ticker_df['Volume'] / ticker_df['Volume_MA20']
            # ATR
            ticker_df['ATR'] = ta.volatility.AverageTrueRange(high=ticker_df['High'], low=ticker_df['Low'], close=ticker_df['Close']).average_true_range()
            # ADX
            adx = ta.trend.ADXIndicator(high=ticker_df['High'], low=ticker_df['Low'], close=ticker_df['Close'])
            ticker_df['ADX'] = adx.adx()
            ticker_df['Ticker'] = ticker
            result.append(ticker_df)
        combined = pd.concat(result)
        combined.set_index('Ticker', append=True, inplace=True)
        combined = combined.reorder_levels(['Ticker', 'Date'])
        return combined