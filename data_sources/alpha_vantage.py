import requests
import pandas as pd
from typing import List, Dict
from core.data_source import DataSource
from core.cache_mixin import CacheMixin
from config.settings import get_settings
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AlphaVantageSource(DataSource, CacheMixin):
    """Data source using Alpha Vantage API (free tier: 5 calls/min, 500 calls/day)."""
    
    def __init__(self, api_key: str = None):
        CacheMixin.__init__(self)
        settings = get_settings()
        self.api_key = api_key or settings.alpha_vantage_api_key
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY in .env")
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_price_data(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily adjusted price data for multiple tickers.
        Note: Free tier limited to 5 calls per minute.
        """
        cache_key = self._cache_key("alpha_price", tickers=sorted(tickers), start=start, end=end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached price data for {len(tickers)} tickers")
            return cached
        
        all_data = []
        for ticker in tqdm(tickers, desc="Fetching from Alpha Vantage"):
            try:
                params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': ticker,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if 'Time Series (Daily)' not in data:
                    logger.warning(f"No data for {ticker}: {data.get('Note', 'Unknown error')}")
                    continue
                
                time_series = data['Time Series (Daily)']
                ticker_data = []
                for date, values in time_series.items():
                    if start <= date <= end:
                        ticker_data.append({
                            'Ticker': ticker,
                            'Date': pd.to_datetime(date),
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['5. adjusted close']),
                            'Volume': int(values['6. volume'])
                        })
                
                if ticker_data:
                    all_data.extend(ticker_data)
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        self._cache.set(cache_key, df, ttl=14400)
        return df
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """Fetch fundamental data (overview)."""
        cache_key = self._cache_key("alpha_fundamental", ticker=ticker)
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            fundamentals = {
                'market_cap': float(data.get('MarketCapitalization', 0)),
                'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio') else 0,
                'revenue_growth': float(data.get('QuarterlyRevenueGrowthYOY', 0)) if data.get('QuarterlyRevenueGrowthYOY') else 0,
                'eps_growth': float(data.get('QuarterlyEarningsGrowthYOY', 0)) if data.get('QuarterlyEarningsGrowthYOY') else 0,
                'sector': data.get('Sector', 'Unknown'),
                'industry': data.get('Industry', 'Unknown')
            }
            
            self._cache.set(cache_key, fundamentals, ttl=86400)
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}
    
    def get_news_headlines(self, ticker: str, lookback_days: int = 7) -> List[str]:
        """
        Alpha Vantage news sentiment endpoint (requires premium tier).
        Falls back to empty list if not available.
        """
        logger.warning(f"News sentiment not available in Alpha Vantage free tier for {ticker}")
        return []
