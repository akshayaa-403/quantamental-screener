import yfinance as yf
from newsapi import NewsApiClient
import os
import pandas as pd
from typing import List, Optional
import re
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
    def get_news_headlines(self, ticker: str, lookback_days: int = 7, company_name: Optional[str] = None,
                           max_headlines: int = 10) -> List[str]:
        """
        Headlines about ``ticker`` published in the last ``lookback_days``.

        NewsAPI (if ``NEWS_API_KEY`` is set) is searched for the company name in
        article *titles*, so short tickers like T, C or V don't match unrelated
        text; without a name, tickers shorter than 3 letters skip NewsAPI.
        Yahoo Finance news (already keyed to the ticker) is the fallback, also
        filtered by publish date. Cached for 1 hour.
        """
        cache_key = self._cache_key("news", ticker=ticker, lookback=lookback_days, name=company_name,
                                    n=max_headlines)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        headlines: List[str] = []

        api_key = os.environ.get('NEWS_API_KEY')
        query = _news_query(ticker, company_name)
        if api_key and query:
            try:
                articles = NewsApiClient(api_key=api_key).get_everything(
                    qintitle=query,
                    from_param=since.strftime('%Y-%m-%dT%H:%M:%S'),
                    language='en',
                    sort_by='publishedAt',
                    page_size=max_headlines,
                ).get('articles', [])
                headlines = _dedupe(a.get('title') for a in articles)[:max_headlines]
                logger.info(f"Fetched {len(headlines)} headlines for {ticker} from NewsAPI.")
            except Exception as e:
                logger.warning(f"NewsAPI failed for {ticker}: {e}. Falling back to yfinance news.")

        if not headlines:
            try:
                recent = []
                for item in yf.Ticker(ticker).news or []:
                    title, published = _yahoo_item(item)
                    if title and (published is None or published >= since):
                        recent.append(title)
                headlines = _dedupe(recent)[:max_headlines]
                logger.info(f"Fetched {len(headlines)} recent headlines for {ticker} from yfinance news.")
            except Exception as e:
                logger.warning(f"yfinance news fallback failed for {ticker}: {e}")

        # Cache even if empty to avoid repeated failed requests
        self._cache.set(cache_key, headlines, ttl=3600)
        return headlines


_NAME_SUFFIX = re.compile(
    r"(\s*\([^)]*\)|,?\s+(inc|incorporated|corp|corporation|co|company|plc|ltd|limited|holdings?|group|"
    r"n\.?v|s\.?a)\.?)$",
    re.IGNORECASE,
)


def _news_query(ticker: str, company_name: Optional[str]) -> Optional[str]:
    """Quoted company name without legal suffixes ("Apple Inc." -> '"Apple"'), or a long-enough ticker."""
    if company_name:
        name = re.sub(r"^the\s+", "", company_name.strip(), flags=re.IGNORECASE)
        while True:
            shorter = _NAME_SUFFIX.sub("", name).strip(" ,.")
            if shorter == name or not shorter:
                break
            name = shorter
        if name:
            return f'"{name}"'
    return ticker if len(ticker) >= 3 else None


def _dedupe(titles) -> List[str]:
    seen, out = set(), []
    for title in titles:
        title = (title or "").strip()
        if title and title.lower() not in seen and title != "[Removed]":
            seen.add(title.lower())
            out.append(title)
    return out


def _yahoo_item(item: dict):
    """(title, publish time as a UTC Timestamp or None) for old and new yfinance news formats."""
    content = item.get('content') if isinstance(item.get('content'), dict) else item
    title = content.get('title')
    published = None
    if content.get('pubDate') or content.get('displayTime'):
        published = pd.to_datetime(content.get('pubDate') or content.get('displayTime'), utc=True, errors='coerce')
    elif item.get('providerPublishTime'):
        published = pd.to_datetime(item['providerPublishTime'], unit='s', utc=True, errors='coerce')
    return title, (None if published is None or pd.isna(published) else published)
