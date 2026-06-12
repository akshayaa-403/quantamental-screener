import click
import pandas as pd
import numpy as np
import logging
from config.settings import get_settings
from data_sources.yahoo_finance import YahooFinanceSource
from sentiment.ensemble_sentiment import EnsembleSentiment
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from output import ConsoleReporter
from utils.logging_utils import setup_logging
from factors.my_factor import MyFactor

logger = logging.getLogger(__name__)

@click.command()
@click.option('--top-n', default=10, help='Number of top stocks to display')
def cli(top_n):
    setup_logging()
    settings = get_settings()
    
    # Components
    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    all_factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor(), MyFactor()]
    # Filter out factors with zero weight (like MyFactor)
    factors = [f for f in all_factors if f.weight != 0.0]
    output = ConsoleReporter()
    
    # Pipeline
    universe = UniverseSelector()
    collector = DataCollector(data_source)
    scorer = ScoringEngine(factors)
    backtester = BacktestEngine()
    
    # Run
    tickers = universe.select()
    logger.info(f"Selected universe: {tickers[:5]}... ({len(tickers)} total)")
    
    # Collect price data (adjust date range to ensure enough history)
    import datetime
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    price_data = collector.collect(tickers, start_date, end_date)
    
    # Sentiment (only for first 20 tickers to avoid rate limits)
    tickers_for_sentiment = tickers[:20] if len(tickers) > 20 else tickers
    logger.info(f"Fetching news for {len(tickers_for_sentiment)} tickers...")
    news_by_ticker = {}
    for t in tickers_for_sentiment:
        headlines = data_source.get_news_headlines(t)
        if headlines:
            news_by_ticker[t] = headlines
        else:
            news_by_ticker[t] = []  # will be ignored
    
    sentiment_scores = sentiment_model.batch_analyze(news_by_ticker)
    
    # Force mock sentiment for demonstration (since real news is sparse)
    if True:  # Set to False after pipeline works
        logger.warning("Using MOCK sentiment scores for demonstration.")
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        sentiment_scores = {
            t: {'score': np.random.uniform(-0.8, 0.8)}
            for t in tickers_in_data
        }
    
    # Score
    scores_df = scorer.compute(price_data, sentiment_scores)
    
    # Backtest (still placeholder)
    backtest_results = backtester.run(scores_df, price_data)
    
    # Output
    output.display_results(
        scores_df.head(top_n),
        backtest_results,
        sector_dist=pd.Series(),
        factor_corr=pd.DataFrame()
    )
    
if __name__ == "__main__":
    cli()