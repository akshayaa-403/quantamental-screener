#!/usr/bin/env python
"""Quick test to verify core functionality without expensive data downloads."""

import pandas as pd
import numpy as np
from config.settings import get_settings
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from factors.my_factor import MyFactor
from pipeline.scoring_engine import ScoringEngine
from pipeline.backtest_engine import BacktestEngine
from output.console_reporter import ConsoleReporter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data():
    """Create mock price data for testing."""
    dates = pd.date_range('2024-01-01', periods=252)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    data = []
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        prices = 100 + np.cumsum(np.random.randn(252) * 2)
        
        for i, date in enumerate(dates):
            data.append({
                'Ticker': ticker,
                'Date': date,
                'Open': prices[i],
                'High': prices[i] + abs(np.random.randn()),
                'Low': prices[i] - abs(np.random.randn()),
                'Close': prices[i],
                'Volume': np.random.randint(1000000, 10000000),
            })
    
    df = pd.DataFrame(data)
    df.set_index(['Ticker', 'Date'], inplace=True)
    return df

def main():
    logger.info("Starting quick test...")
    
    # Test config
    settings = get_settings()
    logger.info(f"Settings loaded: Redis enabled = {settings.redis.enabled}")
    
    # Create mock data
    logger.info("Creating mock price data...")
    price_data = create_mock_data()
    logger.info(f"Mock data shape: {price_data.shape}")
    
    # Test factors
    logger.info("Testing factors...")
    factors = [
        MomentumFactor(),
        SentimentFactor(),
        VolumeFactor(),
        VolatilityFactor(),
        MyFactor()
    ]
    
    # Test sentiment scores
    sentiment_scores = {
        'AAPL': {'score': 0.3},
        'MSFT': {'score': 0.4},
        'GOOGL': {'score': 0.2},
        'AMZN': {'score': 0.1},
        'NVDA': {'score': 0.5},
    }
    
    # Score
    logger.info("Computing scores...")
    scorer = ScoringEngine(factors)
    scores_df = scorer.compute(price_data, sentiment_scores=sentiment_scores)
    logger.info(f"Scores:\n{scores_df}")
    
    # Backtest
    logger.info("Running backtest...")
    backtester = BacktestEngine()
    backtest_results = backtester.run(scores_df, price_data)
    logger.info(f"Backtest results: {backtest_results}")
    
    # Output
    logger.info("Displaying results...")
    output = ConsoleReporter()
    output.display_results(
        scores_df.head(3),
        backtest_results,
        sector_dist=pd.Series(),
        factor_corr=pd.DataFrame()
    )
    
    logger.info("✓ Quick test passed!")
    return 0

if __name__ == "__main__":
    exit(main())
