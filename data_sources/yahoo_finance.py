import yfinance as yf
from newsapi import NewsApiClient
import os
import pandas as pd
from typing import List
from core.cache_mixin import CacheMixin
from tqdm import tqdm
import logging
import requests
from utils.logging_utils import retry_on_network_error

logger = logging.getLogger(__name__)

class YahooFinanceSource(CacheMixin):
    def __init__(self):
        CacheMixin.__init__(self)

    @retry_on_network_error
    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV from Yahoo Finance, indexed by (Ticker, Date).
        Prices are split- and dividend-adjusted, so returns are total returns
        like the benchmark's. Retries up to 3 times on network errors.
        """
        all_dfs = []

        for ticker in tqdm(tickers, desc="Downloading price data"):
            try:
                df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                df = df.reset_index()
                df['Ticker'] = ticker
                all_dfs.append(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']])

            except Exception as e:
                # Network errors restart the whole download via the retry
                # decorator; anything else skips this ticker.
                if isinstance(e, (requests.RequestException, ConnectionError, TimeoutError)):
                    raise
                logger.error(f"Failed to download {ticker}: {e}", exc_info=True)

        if not all_dfs:
            logger.error("No price data downloaded for any ticker.")
            return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])

        combined = pd.concat(all_dfs, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values(['Ticker', 'Date'])
        combined.set_index(['Ticker', 'Date'], inplace=True)

        logger.info(f"Success: {len(combined.index.get_level_values('Ticker').unique())} tickers, shape {combined.shape}")
        return combined

    @retry_on_network_error
    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        """
        Fetch news headlines for a ticker.
        First tries NewsAPI, falls back to yfinance news if that fails.
        Cached for 1 hour.
        """
        cache_key = self._cache_key("news", ticker=ticker, lookback=lookback_days)
        cached = self._cache.get(cache_key)
        if cached is not None:
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
