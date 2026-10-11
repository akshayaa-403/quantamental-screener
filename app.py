import logging
import os
import datetime

from dotenv import load_dotenv

# Load .env BEFORE importing settings/modules: settings read environment
# variables at import time, so .env values (FACTOR_*, UNIVERSE_*, API keys,
# USE_MOCK_SENTIMENT, ...) must be present in os.environ first.
load_dotenv()

import click
import numpy as np
from config.settings import get_settings, period_to_days, WARMUP_DAYS
from data_sources.yahoo_finance import YahooFinanceSource
from sentiment.ensemble_sentiment import EnsembleSentiment
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from output import ConsoleReporter
from utils.logging_utils import setup_logging

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
    factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor()]
    output = ConsoleReporter()

    universe = UniverseSelector()
    collector = DataCollector(data_source)
    scorer = ScoringEngine(factors, weights=factor_weights)
    backtester = BacktestEngine()  # holds BACKTEST_TOP_N_STOCKS; --top-n is display only

    tickers = universe.select()
    logger.info(f"Selected universe: {tickers[:5]}... ({len(tickers)} total)")

    # The backtest window is BACKTEST_PERIOD; WARMUP_DAYS more history before
    # it lets 12-minus-1 momentum (252 trading days) score from day one.
    today = datetime.date.today()
    backtest_start = today - datetime.timedelta(days=period_to_days(settings.backtest.period))
    data_start = backtest_start - datetime.timedelta(days=WARMUP_DAYS)
    price_data = collector.collect(tickers, data_start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

    # Mock sentiment is opt-in (for demos/offline runs without a news API) via
    # USE_MOCK_SENTIMENT=true, seeded so runs are reproducible.
    if os.getenv('USE_MOCK_SENTIMENT', 'false').lower() == 'true':
        logger.warning("USE_MOCK_SENTIMENT=true: using seeded random mock sentiment (demo mode).")
        rng = np.random.default_rng(42)
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        sentiment_scores = {
            t: {'score': float(rng.uniform(-0.8, 0.8))}
            for t in tickers_in_data
        }
    else:
        # Real sentiment for the live ranking only (first 20 names, to stay
        # inside news API rate limits). Names without news are left out and
        # scored as neutral by SentimentFactor.
        news_by_ticker = {}
        for t in tickers[:20]:
            try:
                headlines = data_source.get_news_headlines(t)
                if headlines:
                    news_by_ticker[t] = headlines
            except Exception as e:
                logger.warning(f"Could not fetch news for {t}: {e}")
        sentiment_scores = sentiment_model.batch_analyze(news_by_ticker)
        if not sentiment_scores:
            logger.warning("No real sentiment scores available; sentiment factor will contribute neutral scores.")

    # Backtest without sentiment: these scores describe today's news and would
    # be lookahead on every past date.
    backtest_results = backtester.run(scorer, price_data, sentiment_scores=None,
                                      start_date=backtest_start.strftime('%Y-%m-%d'))
    if not backtest_results:
        logger.error("Backtest returned empty results.")
        return

    # Compute current scores for display (latest date)
    scores_df = scorer.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
    if scores_df.empty:
        logger.error("No scores computed.")
        return

    last_date = scores_df.index.get_level_values('Date').max()
    output.display_results(scores_df.xs(last_date, level='Date'), backtest_results, top_n)

if __name__ == "__main__":
    cli()