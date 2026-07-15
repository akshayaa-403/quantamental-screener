from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class RedisSettings(BaseSettings):
    """Redis configuration - reads from REDIS_* environment variables"""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = "redis://localhost:6379/0"
    enabled: bool = False
    ttl_default: int = 3600
    ttl_price: int = 14400      # 4 hours
    ttl_sentiment: int = 604800 # 7 days
    ttl_factor: int = 86400     # 1 day

class UniverseSettings(BaseSettings):
    """Universe configuration - reads from UNIVERSE_* environment variables"""
    model_config = SettingsConfigDict(env_prefix="UNIVERSE_")
    
    size: int = 50
    min_market_cap: float = 5e9
    min_volume: float = 1e6

class FactorWeights(BaseSettings):
    """Factor weight configuration - reads from FACTOR_* environment variables"""
    model_config = SettingsConfigDict(env_prefix="FACTOR_")
    
    momentum: float = 0.4
    sentiment: float = 0.3
    volume: float = 0.2
    volatility: float = 0.1

class BacktestSettings(BaseSettings):
    """Backtest configuration - reads from BACKTEST_* environment variables"""
    model_config = SettingsConfigDict(env_prefix="BACKTEST_")
    
    period: str = "1y"
    rebalance_freq: int = 5  # days
    top_n_stocks: int = 10
    risk_free_rate: float = 0.0  # annualized; used in the Sharpe ratio

class Settings(BaseSettings):
    """Main settings class that aggregates all sub-settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    redis: RedisSettings = RedisSettings()
    universe: UniverseSettings = UniverseSettings()
    weights: FactorWeights = FactorWeights()
    backtest: BacktestSettings = BacktestSettings()
    
    finnhub_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None

@lru_cache()
def get_settings() -> Settings:
    return Settings()