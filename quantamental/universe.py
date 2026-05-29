"""
Universe selection – retrieves and filters S&P 500 stocks based on market cap, volume, and tradability.
"""
from typing import List, Dict
import pandas as pd
import yfinance as yf
from config.settings import Config


class UniverseSelector:
    """Selects and manages the stock universe"""

    def __init__(self, config: Config):
        self.config = config
        self.universe = []
        self.sector_info = {}

    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_df = tables[0]

            # Store sector information
            for _, row in sp500_df.iterrows():
                self.sector_info[row['Symbol']] = {
                    'sector': row['GICS Sector'],
                    'industry': row['GICS Sub-Industry'],
                    'company_name': row['Security']
                }

            return sp500_df['Symbol'].tolist()
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            # Fallback to predefined list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

    def filter_universe(self, tickers: List[str]) -> List[str]:
        """Filter universe based on market cap and volume criteria"""
        filtered_tickers = []
        print(f"Filtering {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers[:self.config.UNIVERSE_SIZE * 2]):
            if i % 10 == 0:
                print(f"   Processing {i}/{min(len(tickers), self.config.UNIVERSE_SIZE * 2)}")

            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Check market cap
                market_cap = info.get('marketCap', 0)
                if market_cap < self.config.MIN_MARKET_CAP:
                    continue

                # Check average volume
                avg_volume = info.get('averageVolume', 0)
                if avg_volume < self.config.MIN_VOLUME:
                    continue

                # Check if stock is tradeable
                if info.get('quoteType') != 'EQUITY':
                    continue

                filtered_tickers.append(ticker)

                if len(filtered_tickers) >= self.config.UNIVERSE_SIZE:
                    break

            except Exception:
                continue

        self.universe = filtered_tickers
        print(f"Universe filtered to {len(self.universe)} stocks")
        return self.universe

    def get_universe(self) -> List[str]:
        """Get the complete filtered universe"""
        if not self.universe:
            sp500_tickers = self.get_sp500_tickers()
            self.filter_universe(sp500_tickers)
        return self.universe

    def get_sector_info(self, ticker: str) -> Dict:
        """Get sector information for a ticker"""
        return self.sector_info.get(ticker, {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'company_name': ticker
        })