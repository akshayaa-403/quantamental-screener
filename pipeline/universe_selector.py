import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from config.settings import get_settings
import logging
import yfinance as yf
import io

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
        tickers = self._fetch_sp500_csv() or self._fetch_fallback_tickers()
        # Yahoo spells share classes with a dash: BRK.B -> BRK-B.
        tickers = [t.replace('.', '-') for t in tickers]

        selected = self._apply_filters(tickers)[:self.universe_size]
        logger.info(f"Universe filtered to {len(selected)} stocks")
        return selected

    def _apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Keep names above ``min_market_cap`` and ``min_volume`` (average daily
        share volume), largest market cap first, so the universe is the biggest
        liquid names rather than whatever the source list happens to put first.

        These are *today's* market caps and *today's* index members, so a
        backtest over this universe has survivorship bias: companies that fell
        out of the index can never be picked. The backtest applies its own
        point-in-time liquidity screen, but a clean fix needs historical
        constituent lists, which no free source here provides.
        """
        if not self.min_market_cap and not self.min_volume:
            return tickers
        fundamentals = self._fetch_fundamentals(tickers)
        if not fundamentals:
            logger.warning("No market cap/volume data available; universe left unfiltered.")
            return tickers
        passing = [
            t for t, (cap, vol) in fundamentals.items()
            if cap is not None and vol is not None
            and cap >= self.min_market_cap and vol >= self.min_volume
        ]
        logger.info(f"{len(passing)}/{len(tickers)} tickers pass market cap >= {self.min_market_cap:,.0f} "
                    f"and volume >= {self.min_volume:,.0f}")
        return sorted(passing, key=lambda t: fundamentals[t][0], reverse=True)

    def _fetch_fundamentals(self, tickers: List[str]) -> Dict[str, tuple]:
        """Return ticker -> (market cap, 3-month average volume); unknowns are None."""
        def one(ticker: str):
            try:
                info = yf.Ticker(ticker).fast_info
                return ticker, (_number(info.get('marketCap')), _number(info.get('threeMonthAverageVolume')))
            except Exception as e:
                logger.debug(f"No fundamentals for {ticker}: {e}")
                return ticker, (None, None)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = dict(pool.map(one, tickers))
        if all(v == (None, None) for v in results.values()):
            return {}
        return results

    def _fetch_sp500_csv(self) -> List[str]:
        """S&P 500 constituents from the maintained CSV on GitHub."""
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].dropna().astype(str).str.strip().tolist()
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from CSV")
                return tickers
        except Exception as e:
            logger.warning(f"S&P 500 CSV fetch failed: {e}")
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

def _number(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value
