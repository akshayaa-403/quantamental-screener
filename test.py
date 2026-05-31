from config.settings import config
from quantamental.universe import UniverseSelector
selector = UniverseSelector(config)
tickers = selector.get_sp500_tickers()
print(f"Fetched {len(tickers)} tickers. Sample: {tickers[:5]}")