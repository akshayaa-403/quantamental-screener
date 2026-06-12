from typing import List
import pandas as pd
import numpy as np
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
        
        # Validate each ticker has enough data
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        min_required = 21
        for ticker in tickers_in_data:
            ticker_rows = price_data.xs(ticker, level='Ticker')
            if len(ticker_rows) < min_required:
                logger.warning(f"Ticker {ticker} has only {len(ticker_rows)} rows (< {min_required}). Factor scores may be zero.")
        
        self._cache.set(cache_key, price_data, ttl=14400)
        return price_data

    def _add_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        # Ensure we have a 'Close' column (case-insensitive)
        close_col = None
        for col in price_data.columns:
            if col.lower() == 'close':
                close_col = col
                break
        if close_col is None:
            logger.error("No 'Close' column found in price data. Cannot add indicators.")
            return price_data

        # Get all unique tickers from the MultiIndex
        tickers = price_data.index.get_level_values('Ticker').unique()
        
        for ticker in tickers:
            try:
                # Method 1: Use boolean mask for reliability
                mask = price_data.index.get_level_values('Ticker') == ticker
                ticker_df = price_data[mask].copy()
                
                # If the result is a Series (only one row), convert to DataFrame
                if isinstance(ticker_df, pd.Series):
                    ticker_df = ticker_df.to_frame().T
                
                # Ensure columns are lowercase for ta compatibility
                ticker_df.columns = [col.lower() for col in ticker_df.columns]
                
                # Get close series - ensure it's 1D
                if 'close' not in ticker_df.columns:
                    logger.warning(f"No 'close' column for {ticker} after renaming")
                    continue
                
                close_series = ticker_df['close']
                # If close_series is a DataFrame (multiple columns), take the first column
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                # Ensure it's a Series (not a numpy array)
                if not isinstance(close_series, pd.Series):
                    close_series = pd.Series(close_series)
                
                # Drop NaN and check length
                close_series = close_series.dropna()
                if len(close_series) < 20:
                    logger.warning(f"{ticker}: insufficient close data ({len(close_series)} rows) for indicators")
                    # Debug: print first few rows of ticker_df to see what's happening
                    if len(ticker_df) > 0:
                        logger.debug(f"{ticker} first few rows: {ticker_df.head(2)}")
                    continue
                
                # Compute indicators (each returns a Series)
                rsi = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
                macd = ta.trend.MACD(close=close_series)
                macd_line = macd.macd()
                macd_signal = macd.macd_signal()
                sma_20 = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
                sma_50 = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()
                
                # Align with the original ticker's index (all rows, not just non-NaN)
                ticker_indices = price_data[mask].index
                # Reindex to the full index, filling missing with NaN
                rsi_aligned = rsi.reindex(ticker_indices)
                macd_line_aligned = macd_line.reindex(ticker_indices)
                macd_signal_aligned = macd_signal.reindex(ticker_indices)
                sma_20_aligned = sma_20.reindex(ticker_indices)
                sma_50_aligned = sma_50.reindex(ticker_indices)
                
                # Store back using the mask
                price_data.loc[ticker_indices, 'RSI'] = rsi_aligned.values
                price_data.loc[ticker_indices, 'MACD'] = macd_line_aligned.values
                price_data.loc[ticker_indices, 'MACD_SIGNAL'] = macd_signal_aligned.values
                price_data.loc[ticker_indices, 'SMA_20'] = sma_20_aligned.values
                price_data.loc[ticker_indices, 'SMA_50'] = sma_50_aligned.values
                
            except Exception as e:
                logger.warning(f"Error adding indicators for {ticker}: {e}", exc_info=False)
        
        return price_data