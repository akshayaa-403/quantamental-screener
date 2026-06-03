import click
from config.settings import get_settings
from data_sources.yahoo_finance import YahooFinanceSource
from sentiment.ensemble_sentiment import EnsembleSentiment
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from output import ConsoleReporter
from utils.logging_utils import setup_logging
import pandas as pd
from factors.my_factor import MyFactor

@click.command()
@click.option('--top-n', default=10, help='Number of top stocks to display')
def cli(top_n):
    setup_logging()
    settings = get_settings()
    
    # Components
    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor(), MyFactor()]
    output = ConsoleReporter()
    
    # Pipeline
    universe = UniverseSelector()
    collector = DataCollector(data_source)
    scorer = ScoringEngine(factors)
    backtester = BacktestEngine()
    
    # Run
    tickers = universe.select()
    price_data = collector.collect(tickers, '2024-01-01', '2025-01-01')
    
    # Sentiment
    news_by_ticker = {t: data_source.get_news_headlines(t) for t in tickers}
    sentiment_scores = sentiment_model.batch_analyze(news_by_ticker)
    
    # Score
    scores_df = scorer.compute(price_data, sentiment_scores)
    
    # Backtest
    backtest_results = backtester.run(scores_df, price_data)
    
    # Output
    output.display_results(scores_df.head(top_n), backtest_results, 
                           sector_dist=pd.Series(), factor_corr=pd.DataFrame())
    
if __name__ == "__main__":
    cli()