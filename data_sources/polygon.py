import requests
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
from core.data_source import DataSource
from core.cache_mixin import CacheMixin
from config.settings import get_settings
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PolygonSource(DataSource, CacheMixin):
    """Data source using Polygon.io API."""
    
    def __init__(self, api_key: str = None):
        CacheMixin.__init__(self)
        settings = get_settings()
        self.api_key = api_key or getattr(settings, 'POLYGON_API_KEY', None)
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY in .env")
        self.base_url = "https://api.polygon.io"
    
    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch aggregate (daily) bars for multiple tickers.
        Note: Free tier limited to previous day data only.
        """
        cache_key = self._cache_key("polygon_price", tickers=sorted(tickers), start=start, end=end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached price data for {len(tickers)} tickers")
            return cached
        
        all_data = []
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        
        for ticker in tqdm(tickers, desc="Fetching from Polygon"):
            try:
                # For free tier, use aggregates endpoint with limited range
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
                params = {'adjusted': 'true', 'limit': 50000, 'apiKey': self.api_key}
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if data.get('status') != 'OK' or 'results' not in data:
                    logger.warning(f"No data for {ticker}: {data.get('error', 'Unknown error')}")
                    continue
                
                for bar in data['results']:
                    all_data.append({
                        'Ticker': ticker,
                        'Date': pd.to_datetime(bar['t'], unit='ms'),
                        'Open': bar['o'],
                        'High': bar['h'],
                        'Low': bar['l'],
                        'Close': bar['c'],
                        'Volume': bar['v']
                    })
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        self._cache.set(cache_key, df, ttl=14400)
        return df
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """Fetch stock financials (free tier limited)."""
        cache_key = self._cache_key("polygon_fundamental", ticker=ticker)
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        # Free tier doesn't include fundamentals. Return mock.
        fundamentals = {
            'market_cap': 0,
            'pe_ratio': 0,
            'revenue_growth': 0,
            'note': 'Fundamentals require paid Polygon plan'
        }
        
        self._cache.set(cache_key, fundamentals, ttl=86400)
        return fundamentals
    
    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        """Fetch news articles (limited in free tier)."""
        cache_key = self._cache_key("polygon_news", ticker=ticker, lookback=lookback_days)
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            url = f"{self.base_url}/v2/reference/news"
            params = {
                'ticker': ticker,
                'published_utc.gte': from_date,
                'limit': 10,
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                headlines = [article.get('title', '') for article in data['results'][:5]]
            else:
                headlines = []
            
            self._cache.set(cache_key, headlines, ttl=3600)
            return headlines
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []