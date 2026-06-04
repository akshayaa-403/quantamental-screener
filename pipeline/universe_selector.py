import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class UniverseSelector:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.universe_size = config.get('size', settings.universe.size)
        self.min_market_cap = config.get('min_market_cap', settings.universe.min_market_cap)
        self.min_volume = config.get('min_volume', settings.universe.min_volume)
    
    def select(self) -> List[str]:
        # Get S&P 500 tickers from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            if not table:
                logger.warning("Could not find table with class 'wikitable', trying ID 'constituents'")
                table = soup.find('table', {'id': 'constituents'})
            
            if not table:
                logger.error("Could not find S&P 500 table, using fallback tickers")
                # Fallback: return common large-cap tickers
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'GS', 'WFC',
                          'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'GILD', 'AMGN', 'JNJ']
                logger.info(f"Using fallback with {len(tickers)} tickers")
                return tickers[:self.universe_size]
            
            tickers = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    tickers.append(ticker)
            
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            # Fallback: return common large-cap tickers
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'GS', 'WFC',
                      'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'GILD', 'AMGN', 'JNJ']
        
        selected = tickers[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected