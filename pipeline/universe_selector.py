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
        tickers = self._fetch_sp500_wikipedia()
        if not tickers:
            tickers = self._fetch_fallback_tickers()
        selected = tickers[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected

    def _fetch_sp500_wikipedia(self) -> List[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            # Parse tables with pandas
            tables = pd.read_html(resp.text)
            for table in tables:
                # Check for a column that might contain ticker symbols (case-insensitive)
                cols = [col.lower() for col in table.columns]
                if 'symbol' in cols or 'ticker' in cols:
                    # Get the actual column name
                    col_name = table.columns[cols.index('symbol')] if 'symbol' in cols else table.columns[cols.index('ticker')]
                    tickers = table[col_name].tolist()
                    # Clean up any whitespace or empty values
                    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
                    logger.info(f"Fetched {len(tickers)} S&P 500 tickers via pandas.read_html")
                    return tickers
            # If pandas didn't find the table, try BeautifulSoup
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            if not table:
                table = soup.find('table', {'class': 'wikitable'})
            if not table:
                logger.warning("Could not find S&P 500 table with BeautifulSoup either")
                return []
            tickers = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    if ticker:
                        tickers.append(ticker)
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers via BeautifulSoup")
            return tickers
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching S&P 500 tickers: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def _fetch_fallback_tickers(self) -> List[str]:
        # Expanded list of 100 common US large‑cap stocks
        base_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'V', 'JNJ',
            'WMT', 'JPM', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'VZ', 'NFLX', 'CSCO',
            'ADBE', 'CRM', 'NKE', 'CMCSA', 'PEP', 'INTC', 'QCOM', 'TXN', 'AVGO', 'AMD',
            'COST', 'CVX', 'WFC', 'BAC', 'XOM', 'KO', 'PFE', 'MRK', 'ABBV', 'ABT',
            'TMO', 'ACN', 'NEE', 'DHR', 'LIN', 'UPS', 'LMT', 'RTX', 'HON', 'GE',
            'IBM', 'GS', 'BLK', 'C', 'SPGI', 'CAT', 'BA', 'AMGN', 'GILD', 'MDT',
            'ISRG', 'SYK', 'BMY', 'LOW', 'T', 'AXP', 'SBUX', 'TGT', 'MU', 'PLD',
            'DE', 'FIS', 'ADI', 'ZTS', 'CCI', 'EQIX', 'PH', 'SHW', 'TEL', 'CL',
            'MET', 'ICE', 'CME', 'DUK', 'SO', 'MDLZ', 'MO', 'AON', 'PYPL', 'CNC',
            'CI', 'CVS', 'EL', 'NOC', 'GD', 'BDX', 'APD', 'ROP', 'VRTX', 'REGN'
        ]
        logger.info(f"Using fallback with {len(base_tickers)} tickers")
        return base_tickers