import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from config.settings import get_settings
import logging
import yfinance as yf
from utils.logging_utils import retry_on_network_error

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.period = config.get('period', settings.backtest.period)
        self.rebalance_freq = config.get('rebalance_freq', settings.backtest.rebalance_freq)
        self.top_n = config.get('top_n', settings.backtest.top_n_stocks)
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.benchmark_ticker = config.get('benchmark', 'SPY')
        # Annualized risk-free rate used in the Sharpe ratio (0.0 = none).
        self.risk_free_rate = config.get('risk_free_rate', settings.backtest.risk_free_rate)

    @retry_on_network_error
    def _fetch_benchmark(self, start_date, end_date) -> pd.Series:
        bench = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
        if bench.empty:
            logger.warning(f"No benchmark data for {self.benchmark_ticker}")
            return pd.Series(dtype=float)

        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)

        if 'Adj Close' in bench.columns:
            price_series = bench['Adj Close']
        elif 'Close' in bench.columns:
            price_series = bench['Close']
        else:
            price_series = bench.iloc[:, 0]
            logger.warning(f"Using '{bench.columns[0]}' as benchmark price")

        if price_series.index.tz is not None:
            price_series.index = price_series.index.tz_localize(None)

        full_range = pd.date_range(start=start_date, end=end_date, freq='B')
        price_series = price_series.reindex(full_range).ffill().bfill()
        return price_series

    def run(self, scoring_engine, price_data: pd.DataFrame, sentiment_scores: Optional[Dict] = None) -> Dict:
        logger.info("Starting backtest engine...")
        if price_data.empty:
            logger.error("No price data provided.")
            return {}

        logger.info(f"Price data shape: {price_data.shape}")
        logger.info(f"Price data index levels: {price_data.index.names}")
        logger.info(f"Price data columns: {price_data.columns.tolist()}")

        # Ensure index is sorted
        price_data = price_data.sort_index()

        # --- FIX: Convert timezone-aware dates to naive ---
        if price_data.index.get_level_values('Date').tz is not None:
            logger.info("Converting price_data Date index to naive datetime")
            price_data = price_data.reset_index()
            price_data['Date'] = price_data['Date'].dt.tz_localize(None)
            price_data = price_data.set_index(['Ticker', 'Date'])
        # -------------------------------------------------

        # If no sentiment, zero out sentiment factor weight
        if sentiment_scores is None:
            for factor in scoring_engine.factors:
                if factor.name == 'sentiment':
                    factor.weight = 0.0
                    logger.info("Sentiment factor weight set to 0 for backtest (no historical sentiment)")
        else:
            logger.info(f"Sentiment scores for {len(sentiment_scores)} tickers")

        # Get all unique dates (trading days) from price_data
        dates = price_data.index.get_level_values('Date').unique().sort_values()
        logger.info(f"Unique dates: {len(dates)} from {dates.min()} to {dates.max()}")
        if len(dates) < 2:
            logger.warning("Not enough dates for backtest.")
            return {}

        # Ensure dates are naive (remove timezone if any) – already done above, but just in case
        if hasattr(dates, 'tz') and dates.tz is not None:
            dates = dates.tz_localize(None)
            logger.info("Converted dates to naive datetime.")

        logger.info("Computing composite scores...")
        scores_df = scoring_engine.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
        logger.info(f"Scores DataFrame shape: {scores_df.shape}")
        if scores_df.empty:
            logger.warning("Scores DataFrame is empty. Backtest will likely fail.")
            return {}

        # Benchmark series for the same date range
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')
        logger.info(f"Fetching benchmark {self.benchmark_ticker} from {start_date} to {end_date}")
        bench_series = self._fetch_benchmark(start_date, end_date)
        logger.info(f"Benchmark series length: {len(bench_series)}")

        bench_aligned = bench_series.reindex(dates).ffill().bfill()
        if bench_aligned.isna().all():
            logger.warning("Benchmark data entirely missing. Using flat 0% return.")
            bench_aligned = pd.Series(1.0, index=dates)

        # Determine rebalance dates
        rebalance_indices = list(range(0, len(dates), self.rebalance_freq))
        if not rebalance_indices:
            rebalance_indices = [0]
        logger.info(f"Rebalance indices (first 10): {rebalance_indices[:10]}...")

        # Portfolio state
        cash = self.initial_capital
        holdings = {}
        portfolio_values = []
        benchmark_values = []

        logger.info(f"Running backtest with {len(rebalance_indices)} rebalance dates, top {self.top_n} stocks")

        # Main loop
        for idx, current_date in enumerate(dates):
            is_rebalance = (idx in rebalance_indices)
            logger.info(f"Loop: idx={idx}, date={current_date}, is_rebalance={is_rebalance}")

            # Get prices for this date
            try:
                day_prices = price_data.xs(current_date, level='Date')['Close']
            except KeyError:
                # No data for this date; carry forward previous value
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])
                    benchmark_values.append(benchmark_values[-1])
                else:
                    portfolio_values.append(cash)
                    benchmark_values.append(cash * (bench_aligned.loc[current_date] / bench_aligned.iloc[0]))
                continue

            # Rebalance if today is a rebalance date
            if is_rebalance:
                logger.info(f"Rebalance on {current_date} (idx={idx})")
                try:
                    date_scores = scores_df.xs(current_date, level='Date')
                except KeyError:
                    logger.warning(f"No scores for {current_date}")
                    pass
                else:
                    date_scores = date_scores.dropna(subset=['composite'])
                    if date_scores.empty:
                        logger.warning(f"Empty scores for {current_date}")
                    else:
                        top_tickers = date_scores.nlargest(self.top_n, 'composite').index.tolist()
                        logger.info(f"Top {len(top_tickers)} tickers: {top_tickers[:5]}...")
                        prices_for_selection = day_prices.reindex(top_tickers).dropna()
                        logger.info(f"Found prices for {len(prices_for_selection)} out of {len(top_tickers)} selected tickers")
                        if not prices_for_selection.empty:
                            # Sell all current holdings
                            if holdings:
                                sell_value = 0.0
                                for ticker, shares in holdings.items():
                                    if ticker in day_prices:
                                        sell_value += shares * day_prices[ticker]
                                cash = sell_value
                                holdings = {}
                                logger.info(f"Sold holdings, cash = {cash:.2f}")
                            # Buy new portfolio (equal-weighted)
                            invest_per_stock = cash / len(prices_for_selection)
                            logger.info(f"Investing {invest_per_stock:.2f} per stock across {len(prices_for_selection)} tickers")
                            for ticker, price in prices_for_selection.items():
                                holdings[ticker] = invest_per_stock / price
                            cash = 0.0
                            logger.info(f"Portfolio rebalanced on {current_date} with {len(holdings)} positions")
                        else:
                            logger.warning(f"No valid prices for selected tickers on {current_date}")

            # Compute current portfolio value
            total_value = cash
            for ticker, shares in holdings.items():
                if ticker in day_prices:
                    total_value += shares * day_prices[ticker]
            portfolio_values.append(total_value)

            # Benchmark value
            bench_value = self.initial_capital * (bench_aligned.loc[current_date] / bench_aligned.iloc[0])
            benchmark_values.append(bench_value)

        logger.info(f"Backtest loop completed. Portfolio length: {len(portfolio_values)}")

        if len(portfolio_values) < 2:
            logger.warning("Backtest generated too few values.")
            return {}

        port_series = pd.Series(portfolio_values, index=dates)
        bench_series = pd.Series(benchmark_values, index=dates)

        port_returns = port_series.pct_change().dropna()
        bench_returns = bench_series.pct_change().dropna()

        total_return = (port_series.iloc[-1] / self.initial_capital) - 1
        benchmark_total_return = (bench_series.iloc[-1] / self.initial_capital) - 1
        excess_return = total_return - benchmark_total_return

        # Annualized Sharpe ratio with a configurable risk-free rate.
        # Sharpe = mean(excess daily return) / std(daily return) * sqrt(252),
        # where the excess return subtracts the per-day risk-free rate.
        rf_daily = self.risk_free_rate / 252
        excess_returns = port_returns - rf_daily
        if port_returns.std() != 0:
            sharpe = excess_returns.mean() / port_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()

        volatility = port_returns.std() * np.sqrt(252)

        excess_daily = port_returns - bench_returns.reindex(port_returns.index).ffill()
        hit_rate = (excess_daily > 0).mean()

        logger.info("Returning backtest results.")
        return {
            'strategy_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'hit_rate': hit_rate,
            'final_value': port_series.iloc[-1],
            'initial_capital': self.initial_capital,
            'portfolio_series': port_series,
            'benchmark_series': bench_series,
        }