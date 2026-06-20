import pandas as pd
import requests
from typing import List
from config.settings import get_settings
import logging
import yfinance as yf
import io
from utils.logging_utils import retry_on_network_error

# Suppress noisy HTTP debug logs
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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
            tickers = self._fetch_sp500_csv_fallback()
        if not tickers:
            tickers = self._fetch_fallback_tickers()

        selected = tickers[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected

    @retry_on_network_error
    def _fetch_sp500_yfinance(self) -> List[str]:
        """Primary: fetch constituents from ^GSPC.components, fallback to SPY holdings."""
        try:
            sp500 = yf.Ticker("^GSPC")
            if hasattr(sp500, 'components') and sp500.components:
                comps = sp500.components
                if isinstance(comps, dict):
                    tickers = list(comps.keys())
                else:
                    tickers = comps
                tickers = [t for t in tickers if t and not t.startswith('^')]
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from ^GSPC.components")
                return tickers
        except Exception as e:
            logger.warning(f"yfinance ^GSPC components failed: {e}")

        try:
            spy = yf.Ticker("SPY")
            holdings = spy.holdings
            if holdings is not None and not holdings.empty:
                if 'Symbol' in holdings.columns:
                    tickers = holdings['Symbol'].dropna().astype(str).str.strip().tolist()
                else:
                    tickers = holdings.index.tolist()
                tickers = [t for t in tickers if t and not t.startswith('^')]
                logger.info(f"Fetched {len(tickers)} tickers from SPY holdings")
                return tickers
        except Exception as e:
            logger.warning(f"yfinance SPY holdings failed: {e}")

        return []

    @retry_on_network_error
    def _fetch_sp500_csv_fallback(self) -> List[str]:
        """Secondary: use the well‑maintained CSV from GitHub."""
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].dropna().astype(str).str.strip().tolist()
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from CSV fallback")
                return tickers
        except Exception as e:
            logger.warning(f"CSV fallback failed: {e}")
        return []

    def _fetch_fallback_tickers(self) -> List[str]:
        """Ultimate hardcoded list of 100 US large‑caps."""
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