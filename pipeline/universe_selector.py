import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List
from config.settings import get_settings
import logging
import yfinance as yf
import io
from utils.logging_utils import retry_on_network_error

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
            tickers = self._fetch_sp500_csv_fallback()
        if not tickers:
            tickers = self._fetch_fallback_tickers()
        selected = tickers[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected

    @retry_on_network_error
    def _fetch_sp500_yfinance(self) -> List[str]:
        """Try SPY holdings (most reliable), then ^GSPC components."""
        try:
            spy = yf.Ticker("SPY")
            holdings = spy.holdings
            if holdings is not None and not holdings.empty:
                if 'Symbol' in holdings.columns:
                    tickers = holdings['Symbol'].dropna().astype(str).str.strip().tolist()
                else:
                    tickers = holdings.index.tolist()
                tickers = [t for t in tickers if t and not t.startswith('^')]
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from SPY holdings")
                return tickers
        except Exception as e:
            logger.warning(f"yfinance SPY holdings failed: {e}")

        try:
            sp500 = yf.Ticker("^GSPC")
            if hasattr(sp500, 'components') and sp500.components:
                comps = sp500.components
                if isinstance(comps, dict):
                    tickers = list(comps.keys())
                else:
                    tickers = comps
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from ^GSPC.components")
                return tickers
        except Exception as e:
            logger.warning(f"yfinance ^GSPC components failed: {e}")
        return []

    @retry_on_network_error
    def _fetch_sp500_wikipedia(self) -> List[str]:
        """Robust Wikipedia parsing with column matching and BeautifulSoup fallback."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)

        for table in tables:
            cols = [str(col).strip().lower() for col in table.columns]
            for idx, col in enumerate(cols):
                if 'symbol' in col or 'ticker' in col:
                    tickers = table.iloc[:, idx].dropna().astype(str).str.strip().tolist()
                    tickers = [t for t in tickers if t and not t.startswith('^')]
                    logger.info(f"Fetched {len(tickers)} S&P 500 tickers via pandas.read_html")
                    return tickers

        # Fallback: BeautifulSoup
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
        return []

    @retry_on_network_error
    def _fetch_sp500_csv_fallback(self) -> List[str]:
        """Fallback to a maintained CSV from GitHub."""
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].dropna().astype(str).str.strip().tolist()
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers from CSV fallback")
            return tickers
        return []

    def _fetch_fallback_tickers(self) -> List[str]:
        """Hardcoded list of 100 US large‑caps as ultimate fallback."""
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