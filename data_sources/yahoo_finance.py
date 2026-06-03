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
        cache_key = self._cache_key("price", tickers=sorted(tickers), start=start, end=end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached price data for {len(tickers)} tickers")
            return cached
        
        data = {}
        for ticker in tqdm(tickers, desc="Downloading price data"):
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                df['Ticker'] = ticker
                data[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")
        combined = pd.concat(data.values())
        combined.reset_index(inplace=True)
        combined.set_index(['Ticker', 'Date'], inplace=True)
        combined.columns = [col.capitalize() for col in combined.columns]
        
        self._cache.set(cache_key, combined, ttl=14400)  # 4 hours
        return combined
    
    def get_fundamentals(self, ticker: str) -> Dict:
        # Basic info from yfinance
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
        
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            headlines = []
        else:
            headlines = [item.get('title', '') for item in news[:5]]
        self._cache.set(cache_key, headlines, ttl=3600)
        return headlines