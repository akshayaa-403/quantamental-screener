from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class RedisSettings(BaseSettings):
    """Redis configuration - reads from REDIS_* environment variables"""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = "redis://localhost:6379/0"
    enabled: bool = False
    ttl_default: int = 3600
    ttl_price: int = 14400      # 4 hours

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
    transaction_cost_bps: float = 10.0  # cost per trade, in basis points of notional
    n_bootstrap: int = 2000  # random portfolios drawn for the significance test

# Calendar days of history fetched before the backtest window for factor
# warm-up: 252 trading days of 12-minus-1 momentum plus a holiday margin.
WARMUP_DAYS = 400

def period_to_days(period: str) -> int:
    """Convert a yfinance-style period ('6mo', '1y', '90d', '2wk') to calendar days."""
    period = period.strip().lower()
    for suffix, days in (('mo', 30), ('wk', 7), ('y', 365), ('d', 1)):
        if period.endswith(suffix):
            return int(float(period[: -len(suffix)]) * days)
    raise ValueError(f"Unrecognized period: {period!r}")

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
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()