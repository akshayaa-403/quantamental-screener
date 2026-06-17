import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from config.settings import get_settings
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

class UniverseSelector:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.universe_size = config.get('size', settings.universe.size)
        self.min_market_cap = config.get('min_market_cap', settings.universe.min_market_cap)
        self.min_volume = config.get('min_volume', settings.universe.min_volume)

    def select(self) -> List[str]:
        tickers = self._fetch_sp500_yfinance()
        if not tickers:
            tickers = self._fetch_sp500_wikipedia()
        if not tickers:
            tickers = self._fetch_fallback_tickers()
        selected = tickers[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected

    def _fetch_sp500_yfinance(self) -> List[str]:
        """Try to get S&P 500 constituents via yfinance (ETF holdings or index components)."""
        try:
            # Attempt 1: ^GSPC may have 'components' if supported
            sp500 = yf.Ticker("^GSPC")
            if hasattr(sp500, 'components') and sp500.components:
                comps = sp500.components
                if isinstance(comps, dict):
                    tickers = list(comps.keys())
                else:
                    tickers = comps
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from ^GSPC.components")
                return tickers

            # Attempt 2: Use SPY ETF holdings (more reliable)
            spy = yf.Ticker("SPY")
            holdings = spy.holdings
            if holdings is not None and not holdings.empty:
                # holdings may have 'Symbol' column or index as symbols
                if 'Symbol' in holdings.columns:
                    tickers = holdings['Symbol'].dropna().tolist()
                else:
                    tickers = holdings.index.tolist()
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from SPY holdings")
                return tickers
        except Exception as e:
            logger.warning(f"yfinance fetch failed: {e}")
        return []

    def _fetch_sp500_wikipedia(self) -> List[str]:
        """Robust Wikipedia parsing using pandas.read_html with column matching."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            for table in tables:
                # Look for a column that contains 'symbol' or 'ticker'
                cols = [col.lower() for col in table.columns]
                if 'symbol' in cols or 'ticker' in cols:
                    col_name = table.columns[cols.index('symbol')] if 'symbol' in cols else table.columns[cols.index('ticker')]
                    tickers = table[col_name].dropna().astype(str).str.strip().tolist()
                    tickers = [t for t in tickers if t and not t.startswith('^')]
                    logger.info(f"Fetched {len(tickers)} S&P 500 tickers via pandas.read_html")
                    return tickers
            # Fallback: BeautifulSoup on the first wikitable
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            if table:
                tickers = []
                for row in table.find_all('tr')[1:]:
                    cells = row.find_all('td')
                    if cells:
                        ticker = cells[0].text.strip()
                        if ticker and not ticker.startswith('^'):
                            tickers.append(ticker)
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers via BeautifulSoup")
                return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        return []

    def _fetch_fallback_tickers(self) -> List[str]:
        """Fallback hardcoded list of 100 US large‑caps."""
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