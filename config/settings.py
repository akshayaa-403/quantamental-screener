"""
Configuration settings for the Quantamental Equity Screener
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration class for the screener"""

    # Universe settings
    UNIVERSE_SIZE: int = 50          # Number of stocks to analyze
    MIN_MARKET_CAP: float = 5e9      # $5B minimum market cap
    MIN_VOLUME: float = 1e6          # $1M average daily volume

    # Technical analysis settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0

    # Sentiment analysis settings
    MAX_NEWS_PER_STOCK: int = 5
    SENTIMENT_LOOKBACK_DAYS: int = 7

    # Scoring weights
    MOMENTUM_WEIGHT: float = 0.4
    SENTIMENT_WEIGHT: float = 0.3
    VOLUME_WEIGHT: float = 0.2
    VOLATILITY_WEIGHT: float = 0.1

    # Risk management
    MAX_POSITION_SIZE: float = 0.05   # 5% max per position
    MAX_SECTOR_EXPOSURE: float = 0.25 # 25% max per sector

    # Backtesting settings
    BACKTEST_PERIOD: str = "1y"
    REBALANCE_FREQ: int = 5           # Rebalance every N days
    TOP_N_STOCKS: int = 10

    # Data sources
    DATA_PERIOD: str = "6mo"
    NEWS_SOURCES: List[str] = None

    def __post_init__(self):
        if self.NEWS_SOURCES is None:
            self.NEWS_SOURCES = ['reuters', 'bloomberg', 'marketwatch', 'yahoo']


# Singleton instance for easy import
config = Config()