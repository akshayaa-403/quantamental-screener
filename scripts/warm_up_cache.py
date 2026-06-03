from data_sources.yahoo_finance import YahooFinanceSource
from config.settings import get_settings

if __name__ == "__main__":
    source = YahooFinanceSource()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    source.get_price_data(tickers, '2023-01-01', '2024-01-01')
    print("Cache warmed up.")