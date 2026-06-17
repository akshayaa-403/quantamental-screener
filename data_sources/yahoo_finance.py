import yfinance as yf
from newsapi import NewsApiClient
import os
import pandas as pd
from typing import List, Dict
from core.data_source import DataSource
from core.cache_mixin import CacheMixin
from tqdm import tqdm
import logging
import requests
from utils.logging_utils import retry_on_network_error

logger = logging.getLogger(__name__)

class YahooFinanceSource(DataSource, CacheMixin):
    def __init__(self):
        CacheMixin.__init__(self)

    @retry_on_network_error
    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch price data for multiple tickers from Yahoo Finance.
        Retries up to 3 times on network errors.
        """
        all_dfs = []
        min_required_days = 21

        for ticker in tqdm(tickers, desc="Downloading price data"):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end, auto_adjust=False)
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                if len(df) < min_required_days:
                    # Try extending start date to get more data
                    extended_start = pd.to_datetime(start) - pd.DateOffset(months=3)
                    df = stock.history(start=extended_start.strftime('%Y-%m-%d'), end=end, auto_adjust=False)
                    if len(df) < min_required_days:
                        logger.warning(f"{ticker} insufficient data ({len(df)} rows). Skipping.")
                        continue

                df = df.reset_index()
                # Standardize column names
                df.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }, inplace=True)

                df['Ticker'] = ticker
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
                all_dfs.append(df)

            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}", exc_info=True)
                # Let the retry decorator handle it if it's a network issue,
                # but we might still want to continue with other tickers.
                # Since we catch all, we don't re-raise; we just skip this ticker.
                # If we want the whole operation to retry on network errors, 
                # we should not catch requests exceptions here.
                # Instead, we catch only non-network errors.
                # For simplicity, we'll re-raise network-related exceptions
                # so that the retry decorator can act on the whole method.
                if isinstance(e, (requests.RequestException, ConnectionError, TimeoutError)):
                    raise
                # Otherwise, log and continue

        if not all_dfs:
            logger.error("No price data downloaded for any ticker.")
            return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])

        combined = pd.concat(all_dfs, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values(['Ticker', 'Date'])
        combined.set_index(['Ticker', 'Date'], inplace=True)

        logger.info(f"Success: {len(combined.index.get_level_values('Ticker').unique())} tickers, shape {combined.shape}")
        return combined

    def get_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch fundamental data from Yahoo Finance info.
        Cached for 24 hours.
        """
        cache_key = self._cache_key("fundamental", ticker=ticker)
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
            self._cache.set(cache_key, fundamentals, ttl=86400)
            return fundamentals
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}", exc_info=True)
            return {}

    @retry_on_network_error
    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        """
        Fetch news headlines for a ticker.
        First tries NewsAPI, falls back to yfinance news if that fails.
        Cached for 1 hour.
        """
        cache_key = self._cache_key("news", ticker=ticker, lookback=lookback_days)
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        headlines = []

        # Try NewsAPI first
        api_key = os.environ.get('NEWS_API_KEY')
        if api_key:
            try:
                newsapi = NewsApiClient(api_key=api_key)
                all_articles = newsapi.get_everything(
                    q=ticker,
                    language='en',
                    sort_by='relevancy',
                    page_size=5
                )
                headlines = [article['title'] for article in all_articles.get('articles', [])]
                logger.info(f"Fetched {len(headlines)} headlines for {ticker} from NewsAPI.")
                if headlines:
                    self._cache.set(cache_key, headlines, ttl=3600)
                    return headlines
            except Exception as e:
                logger.warning(f"NewsAPI failed for {ticker}: {e}. Falling back to yfinance news.")

        # Fallback to yfinance news if NewsAPI returned nothing or failed
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                headlines = [item.get('title', '') for item in news[:5] if item.get('title')]
                logger.info(f"Fetched {len(headlines)} headlines for {ticker} from yfinance news.")
            else:
                logger.info(f"No news found for {ticker} via yfinance.")
        except Exception as e:
            logger.warning(f"yfinance news fallback failed for {ticker}: {e}")

        # Cache even if empty to avoid repeated failed requests
        self._cache.set(cache_key, headlines, ttl=3600)
        return headlines