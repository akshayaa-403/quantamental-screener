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
        try:
            import requests
            from bs4 import BeautifulSoup
        
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        
            # Parse with BeautifulSoup to find the correct table
            soup = BeautifulSoup(response.text, 'lxml')
            # The main table has class "wikitable sortable" or just "wikitable"
            table = soup.find('table', class_='wikitable')
            if not table:
                # Fallback: try to find any table with "wikitable"
                table = soup.find('table', {'class': 'wikitable'})
        
            if table:
                # Convert the table HTML to a pandas DataFrame
                df = pd.read_html(str(table))[0]
            else:
                # If no table found, fallback to reading all tables (old method)
                tables = pd.read_html(response.text)
                if not tables:
                    raise ValueError("No tables found on Wikipedia page")
                df = tables[0]
        
        # The first column is usually 'Symbol'
        if 'Symbol' not in df.columns:
            # Try to find the column by position (first column)
            symbol_col = df.columns[0]
        else:
            symbol_col = 'Symbol'
        
        tickers = df[symbol_col].tolist()
        
        # Store sector information (adjust column names as needed)
        # Typical columns: 'Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry'
        sector_col = 'GICS Sector' if 'GICS Sector' in df.columns else None
        industry_col = 'GICS Sub-Industry' if 'GICS Sub-Industry' in df.columns else None
        name_col = 'Security' if 'Security' in df.columns else None
        
        for _, row in df.iterrows():
            ticker = row[symbol_col]
            self.sector_info[ticker] = {
                'sector': row[sector_col] if sector_col else 'Unknown',
                'industry': row[industry_col] if industry_col else 'Unknown',
                'company_name': row[name_col] if name_col else ticker
            }
        
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers

    except ImportError as e:
        if 'lxml' in str(e).lower() or 'bs4' in str(e).lower():
            print("Error: Missing required library. Please run: pip install lxml beautifulsoup4")
        else:
            print(f"Import error: {e}")
        print("Using fallback predefined ticker list.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

    except Exception as e:
        print(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        print("Using fallback predefined ticker list.")
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

                market_cap = info.get('marketCap', 0)
                if market_cap < self.config.MIN_MARKET_CAP:
                    continue

                avg_volume = info.get('averageVolume', 0)
                if avg_volume < self.config.MIN_VOLUME:
                    continue

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