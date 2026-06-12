import yfinance as yf
import pandas as pd
from typing import List, Dict
from core.data_source import DataSource
from core.cache_mixin import CacheMixin
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class YahooFinanceSource(DataSource, CacheMixin):
    def __init__(self):
        CacheMixin.__init__(self)

    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        all_dfs = []
        min_required_days = 21

        for ticker in tqdm(tickers, desc="Downloading price data"):
            try:
                # Use Ticker.history() which returns a simple DataFrame (no MultiIndex)
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end, auto_adjust=False)
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                if len(df) < min_required_days:
                    # Try to get more data by extending start date
                    extended_start = pd.to_datetime(start) - pd.DateOffset(months=3)
                    df = stock.history(start=extended_start.strftime('%Y-%m-%d'), end=end, auto_adjust=False)
                    if len(df) < min_required_days:
                        logger.warning(f"{ticker} insufficient data ({len(df)} rows). Skipping.")
                        continue

                # Reset index to make Date a column
                df = df.reset_index()
                # Ensure column names are standard (already Open, High, Low, Close, Volume)
                df.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }, inplace=True)

                # Add ticker column
                df['Ticker'] = ticker

                # Keep only needed columns
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]

                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")

        if not all_dfs:
            logger.error("No price data downloaded")
            return pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume'])

        combined = pd.concat(all_dfs, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values(['Ticker', 'Date'])
        combined.set_index(['Ticker', 'Date'], inplace=True)

        logger.info(f"Success: {len(combined.index.get_level_values('Ticker').unique())} tickers, shape {combined.shape}")
        return combined

    def get_fundamentals(self, ticker: str) -> Dict:
        cache_key = self._cache_key("fundamental", ticker=ticker)
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
        }
        self._cache.set(cache_key, fundamentals, ttl=86400)
        return fundamentals

    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        cache_key = self._cache_key("news", ticker=ticker, lookback=lookback_days)
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if not news:
                headlines = []
            else:
                headlines = [item.get('title', '') for item in news[:5] if item.get('title')]
        except Exception as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            headlines = []

        self._cache.set(cache_key, headlines, ttl=3600)
        return headlines