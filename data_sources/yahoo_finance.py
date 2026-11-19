import yfinance as yf
from newsapi import NewsApiClient
import os
import pandas as pd
from typing import Dict, List, Optional
from urllib.parse import urlparse
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

    def get_news_headlines(self, ticker: str, lookback_days: int = 7, company_name: Optional[str] = None,
                           max_headlines: int = 10) -> List[str]:
        """Titles of :meth:`get_news_articles` (what the sentiment model scores)."""
        return [a['title'] for a in self.get_news_articles(ticker, lookback_days, company_name, max_headlines)]

    @retry_on_network_error
    def get_news_articles(self, ticker: str, lookback_days: int = 7, company_name: Optional[str] = None,
                          max_headlines: int = 10) -> List[Dict[str, Optional[str]]]:
        """
        Articles about ``ticker`` published in the last ``lookback_days``, as
        ``{'title', 'url', 'source', 'tag'}`` dicts (``url``/``source`` may be None;
        ``tag`` is a short outlet label such as "WSJ" or "NYT").

        NewsAPI (if ``NEWS_API_KEY`` is set) is searched for the company name in
        article *titles*, so short tickers like T, C or V don't match unrelated
        text; without a name, tickers shorter than 3 letters skip NewsAPI.
        Yahoo Finance news (already keyed to the ticker) is the fallback, also
        filtered by publish date. Cached for 1 hour.
        """
        cache_key = self._cache_key("news_articles", ticker=ticker, lookback=lookback_days, name=company_name,
                                    n=max_headlines)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        articles: List[Dict[str, Optional[str]]] = []

        api_key = os.environ.get('NEWS_API_KEY')
        query = _news_query(ticker, company_name)
        if api_key and query:
            try:
                raw = NewsApiClient(api_key=api_key).get_everything(
                    qintitle=query,
                    from_param=since.strftime('%Y-%m-%dT%H:%M:%S'),
                    language='en',
                    sort_by='publishedAt',
                    page_size=max_headlines,
                ).get('articles', [])
                articles = _dedupe(_article(a.get('title'), a.get('url'), (a.get('source') or {}).get('name'))
                                   for a in raw)[:max_headlines]
                logger.info(f"Fetched {len(articles)} headlines for {ticker} from NewsAPI.")
            except Exception as e:
                logger.warning(f"NewsAPI failed for {ticker}: {e}. Falling back to yfinance news.")

        if not articles:
            try:
                recent = []
                for item in yf.Ticker(ticker).news or []:
                    article, published = _yahoo_item(item)
                    if article['title'] and (published is None or published >= since):
                        recent.append(article)
                articles = _dedupe(recent)[:max_headlines]
                logger.info(f"Fetched {len(articles)} recent headlines for {ticker} from yfinance news.")
            except Exception as e:
                logger.warning(f"yfinance news fallback failed for {ticker}: {e}")

        # Cache even if empty to avoid repeated failed requests
        self._cache.set(cache_key, articles, ttl=3600)
        return articles


# Short labels for well-known outlets, matched against the lowercased publisher
# name (NewsAPI source / Yahoo provider) and the article URL's domain.
_SOURCE_TAGS = [
    (("new york times", "nytimes.com"), "NYT"),
    (("wall street journal", "wsj.com"), "WSJ"),
    (("financial times", "ft.com"), "FT"),
    (("bloomberg",), "Bloomberg"),
    (("reuters",), "Reuters"),
    (("cnbc",), "CNBC"),
    (("washington post", "washingtonpost.com"), "WaPo"),
    (("associated press", "apnews.com"), "AP"),
    (("barron",), "Barron's"),
    (("marketwatch",), "MarketWatch"),
    (("business insider", "businessinsider.com"), "BI"),
    (("the economist", "economist.com"), "Economist"),
    (("forbes",), "Forbes"),
    (("fortune",), "Fortune"),
    (("motley fool", "fool.com"), "Motley Fool"),
    (("seeking alpha", "seekingalpha.com"), "Seeking Alpha"),
    (("investor's business daily", "investors.com"), "IBD"),
    (("zacks",), "Zacks"),
    (("benzinga",), "Benzinga"),
    (("techcrunch",), "TechCrunch"),
    (("the verge", "theverge.com"), "The Verge"),
    (("bbc",), "BBC"),
    (("guardian",), "Guardian"),
    (("cnn.com", "cnn "), "CNN"),
    (("yahoo",), "Yahoo Finance"),
]


def source_tag(source: Optional[str], url: Optional[str] = None) -> Optional[str]:
    """Short outlet label ("WSJ", "NYT", ...); the publisher name if unknown, else the URL's domain."""
    domain = re.sub(r"^www\.", "", urlparse(url).netloc.lower()) if url else ""
    haystack = f" {(source or '').lower()} {domain} "
    for needles, tag in _SOURCE_TAGS:
        if any(n in haystack for n in needles):
            return tag
    return (source or "").strip() or domain or None


def _article(title, url, source) -> Dict[str, Optional[str]]:
    title = (title or "").strip()
    url = url if isinstance(url, str) and url.startswith(("http://", "https://")) else None
    source = (source or "").strip() or None
    return {'title': title, 'url': url, 'source': source, 'tag': source_tag(source, url)}


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


def _dedupe(articles) -> List[Dict[str, Optional[str]]]:
    """Drop empty, removed and repeated (case-insensitive) titles, keeping the first."""
    seen, out = set(), []
    for article in articles:
        title = article['title']
        if title and title.lower() not in seen and title != "[Removed]":
            seen.add(title.lower())
            out.append(article)
    return out


def _yahoo_item(item: dict):
    """(article dict, publish time as a UTC Timestamp or None) for old and new yfinance news formats."""
    content = item.get('content') if isinstance(item.get('content'), dict) else item
    url = ((content.get('canonicalUrl') or {}).get('url') or (content.get('clickThroughUrl') or {}).get('url')
           or content.get('link') or item.get('link'))
    source = (content.get('provider') or {}).get('displayName') or content.get('publisher') or item.get('publisher')
    article = _article(content.get('title'), url, source)
    published = None
    if content.get('pubDate') or content.get('displayTime'):
        published = pd.to_datetime(content.get('pubDate') or content.get('displayTime'), utc=True, errors='coerce')
    elif item.get('providerPublishTime'):
        published = pd.to_datetime(item['providerPublishTime'], unit='s', utc=True, errors='coerce')
    return article, (None if published is None or pd.isna(published) else published)
