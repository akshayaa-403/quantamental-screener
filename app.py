import sys
sys.stdout = open('nul', 'w')

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
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

@click.command()
@click.option('--top-n', default=10, help='Number of top stocks to display')
def cli(top_n):
    setup_logging()
    settings = get_settings()

    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    all_factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor(), MyFactor()]
    factors = [f for f in all_factors if f.weight != 0.0]
    output = ConsoleReporter()

    universe = UniverseSelector()
    collector = DataCollector(data_source)
    scorer = ScoringEngine(factors)
    backtester = BacktestEngine({'top_n': top_n})

    tickers = universe.select()
    logger.info(f"Selected universe: {tickers[:5]}... ({len(tickers)} total)")

    import datetime
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    price_data = collector.collect(tickers, start_date, end_date)

    # Sentiment (only for first 20 to avoid rate limits)
    tickers_for_sentiment = tickers[:20]
    news_by_ticker = {}
    for t in tickers_for_sentiment:
        try:
            headlines = data_source.get_news_headlines(t)
            if headlines:
                news_by_ticker[t] = headlines
        except Exception as e:
            logger.warning(f"Could not fetch news for {t}: {e}")
    sentiment_scores = sentiment_model.batch_analyze(news_by_ticker)

    # If real sentiment is sparse, we may still use mock for demonstration
    # Set to False once you have reliable news API
    USE_MOCK_SENTIMENT = True
    if USE_MOCK_SENTIMENT:
        logger.warning("Using MOCK sentiment scores for demonstration.")
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        sentiment_scores = {
            t: {'score': np.random.uniform(-0.8, 0.8)}
            for t in tickers_in_data
        }

    # Now run backtest (pass scorer and data)
    backtest_results = backtester.run(scorer, price_data, sentiment_scores)
    if not backtest_results:
        logger.error("Backtest returned empty results.")
        return

    # Also compute current scores for display
    scores_df = scorer.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
    top_scores = scores_df.groupby('Date').last().nlargest(top_n, 'composite')

    output.display_results(
        top_scores,
        backtest_results,
        sector_dist=pd.Series(),
        factor_corr=pd.DataFrame()
    )

if __name__ == "__main__":
    cli()