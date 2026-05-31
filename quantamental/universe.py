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
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable'})
            if not table:
                raise ValueError("Could not find wikitable on page")

            rows = table.find_all('tr')
            if len(rows) < 2:
                raise ValueError("Table has no data rows")

            header_cells = rows[0].find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]

            expected_cols = ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']
            col_indices = {}
            for col in expected_cols:
                try:
                    col_indices[col] = headers.index(col)
                except ValueError:
                    if col == 'Symbol':
                        col_indices[col] = 0
                    elif col == 'Security':
                        col_indices[col] = 1
                    elif col == 'GICS Sector':
                        col_indices[col] = 2
                    elif col == 'GICS Sub-Industry':
                        col_indices[col] = 3

            tickers = []
            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue
                ticker_cell = cells[col_indices['Symbol']]
                ticker_link = ticker_cell.find('a')
                if ticker_link:
                    ticker = ticker_link.get_text(strip=True)
                else:
                    ticker = ticker_cell.get_text(strip=True)

                if not ticker:
                    continue

                security = cells[col_indices['Security']].get_text(strip=True)
                sector = cells[col_indices['GICS Sector']].get_text(strip=True)
                industry = cells[col_indices['GICS Sub-Industry']].get_text(strip=True)

                tickers.append(ticker)
                self.sector_info[ticker] = {
                    'sector': sector,
                    'industry': industry,
                    'company_name': security
                }

            if not tickers:
                raise ValueError("No tickers extracted from table")

            print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers

        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            print("Using fallback predefined ticker list.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

    def filter_universe(self, tickers: List[str]) -> List[str]:
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
        if not self.universe:
            sp500_tickers = self.get_sp500_tickers()
            self.filter_universe(sp500_tickers)
        return self.universe

    def get_sector_info(self, ticker: str) -> Dict:
        return self.sector_info.get(ticker, {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'company_name': ticker
        })