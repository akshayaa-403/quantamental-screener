import logging
import os
import datetime

from dotenv import load_dotenv

# Load .env BEFORE importing settings/modules: settings read environment
# variables at import time, so .env values (FACTOR_*, UNIVERSE_*, API keys,
# USE_MOCK_SENTIMENT, ...) must be present in os.environ first.
load_dotenv()

import click
import pandas as pd
import numpy as np
from config.settings import get_settings
from data_sources.yahoo_finance import YahooFinanceSource
from sentiment.ensemble_sentiment import EnsembleSentiment
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from output import ConsoleReporter
from utils.logging_utils import setup_logging
from factors.my_factor import MyFactor

# Basic logging configuration (refined by setup_logging in cli). Kept quiet at
# import time so `import app` doesn't spam DEBUG output.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--top-n', default=10, help='Number of top stocks to display')
def cli(top_n):
    setup_logging()
    settings = get_settings()

    # Factor weights come from a single source of truth: settings.weights
    # (configurable via FACTOR_* env vars / .env). Previously the CLI passed no
    # weights, so every factor silently defaulted to weight 1.0 (equal weight)
    # and the configured weights were ignored.
    factor_weights = {
        'momentum': settings.weights.momentum,
        'sentiment': settings.weights.sentiment,
        'volume': settings.weights.volume,
        'volatility': settings.weights.volatility,
    }

    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    all_factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor(), MyFactor()]
    # Drop factors with a zero weight (MyFactor is disabled by default).
    factors = [f for f in all_factors if factor_weights.get(f.name, f.weight) != 0.0]
    output = ConsoleReporter()

    universe = UniverseSelector()
    collector = DataCollector(data_source)
    scorer = ScoringEngine(factors, weights=factor_weights)
    backtester = BacktestEngine({'top_n': top_n})

    tickers = universe.select()
    logger.info(f"Selected universe: {tickers[:5]}... ({len(tickers)} total)")

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

    # Mock sentiment is opt-in (for demos/offline runs without a news API) via
    # USE_MOCK_SENTIMENT=true. It is seeded so runs are reproducible. By default
    # we keep the real sentiment scores; tickers without news contribute a
    # neutral 0.0 (handled by SentimentFactor), rather than random noise.
    if os.getenv('USE_MOCK_SENTIMENT', 'false').lower() == 'true':
        logger.warning("USE_MOCK_SENTIMENT=true: overriding with seeded random mock sentiment (demo mode).")
        rng = np.random.default_rng(42)
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        sentiment_scores = {
            t: {'score': float(rng.uniform(-0.8, 0.8))}
            for t in tickers_in_data
        }
    elif not sentiment_scores:
        logger.warning("No real sentiment scores available; sentiment factor will contribute neutral (0.0) scores.")

    # Run backtest
    backtest_results = backtester.run(scorer, price_data, sentiment_scores)
    if not backtest_results:
        logger.error("Backtest returned empty results.")
        return

    # Compute current scores for display (latest date)
    scores_df = scorer.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
    if scores_df.empty:
        logger.error("No scores computed.")
        return

    # Get the most recent date and select top N
    last_date = scores_df.index.get_level_values('Date').max()
    current_scores = scores_df.xs(last_date, level='Date')
    top_scores = current_scores.nlargest(top_n, 'composite')

    # Display results
    output.display_results(
        top_scores,
        backtest_results,
        sector_dist=pd.Series(),
        factor_corr=pd.DataFrame()
    )

if __name__ == "__main__":
    cli()