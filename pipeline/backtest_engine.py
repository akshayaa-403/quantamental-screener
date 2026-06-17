import pandas as pd
import numpy as np
from typing import Dict, Optional
from config.settings import get_settings
import logging
import yfinance as yf
from datetime import timedelta

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.period = config.get('period', settings.backtest.period)
        self.rebalance_freq = config.get('rebalance_freq', settings.backtest.rebalance_freq)  # days
        self.top_n = config.get('top_n', settings.backtest.top_n_stocks)
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.benchmark_ticker = config.get('benchmark', 'SPY')

    def run(self, scoring_engine, price_data: pd.DataFrame, sentiment_scores: Optional[Dict] = None) -> Dict:
        """
        Run backtest using historical factor scores.
        - scoring_engine: instance of ScoringEngine (with factors)
        - price_data: MultiIndex (Ticker, Date) with columns Open, High, Low, Close, Volume
        - sentiment_scores: dict of ticker -> sentiment dict (optional, may be None for backtest)
        """
        # Ensure data is sorted
        price_data = price_data.sort_index()

        # Compute daily composite scores (cross‑sectional normalization per date)
        # We need to pass sentiment_scores if available; if not, we ignore sentiment factor (weight set to 0)
        if sentiment_scores is None:
            # Temporarily zero out sentiment factor weight
            for factor in scoring_engine.factors:
                if factor.name == 'sentiment':
                    factor.weight = 0.0
                    logger.info("Sentiment factor weight set to 0 for backtest (no historical sentiment data)")

        # Compute scores for all dates
        scores_df = scoring_engine.compute(price_data, sentiment_scores, cross_sectional_normalize=True)

        # Get unique dates
        dates = price_data.index.get_level_values('Date').unique().sort_values()
        if len(dates) < 2:
            logger.warning("Not enough dates for backtest.")
            return {}

        # Prepare benchmark data (SPY)
        bench_start = dates.min().strftime('%Y-%m-%d')
        bench_end = dates.max().strftime('%Y-%m-%d')
        bench = yf.download(self.benchmark_ticker, start=bench_start, end=bench_end, progress=False)
        if bench.empty:
            logger.warning(f"Could not fetch benchmark {self.benchmark_ticker}. Using 0% return.")
            bench_returns = pd.Series(0.0, index=dates)
        else:
            bench = bench['Adj Close'].reindex(dates, method='ffill')
            bench_returns = bench.pct_change().fillna(0.0)

        # Rebalance dates: start from the first date, step by rebalance_freq days
        rebalance_dates = dates[::self.rebalance_freq]
        if len(rebalance_dates) == 0:
            rebalance_dates = [dates[0]]

        # Portfolio tracking
        portfolio_value = self.initial_capital
        holdings = {}  # ticker -> shares held
        daily_portfolio_values = []
        daily_benchmark_values = [self.initial_capital]

        # Loop over each rebalance period
        for i, rebal_date in enumerate(rebalance_dates):
            # Determine holding period end: next rebalance date or end of data
            if i + 1 < len(rebalance_dates):
                next_rebal = rebalance_dates[i + 1]
            else:
                next_rebal = dates[-1]

            # On rebalance date, select top N stocks based on composite score at that date
            # We need the scores for that specific date
            date_scores = scores_df.xs(rebal_date, level='Date')
            # Drop NaN scores
            date_scores = date_scores.dropna(subset=['composite'])
            if date_scores.empty:
                logger.warning(f"No scores on {rebal_date}, skipping rebalance")
                continue

            # Rank by composite (descending) and pick top N
            top_tickers = date_scores.nlargest(self.top_n, 'composite').index.tolist()

            # Get prices on rebalance date for these tickers
            prices_at_date = price_data.xs(rebal_date, level='Date')['Close']
            prices_at_date = prices_at_date.reindex(top_tickers).dropna()
            if prices_at_date.empty:
                continue

            # Sell all current holdings (if any) and buy top N
            # Simulate closing positions at current prices
            current_prices = prices_at_date
            if holdings:
                # Compute current value of holdings
                current_value = 0
                for ticker, shares in holdings.items():
                    if ticker in current_prices:
                        current_value += shares * current_prices[ticker]
                    else:
                        # If ticker delisted, value = 0
                        pass
                # Add cash from selling (already reflected in portfolio_value)
                portfolio_value = current_value  # assuming all cash reinvested

            # Now invest in top N equally weighted
            if len(current_prices) == 0:
                continue
            invest_per_stock = portfolio_value / len(current_prices)
            holdings = {}
            for ticker, price in current_prices.items():
                shares = invest_per_stock / price
                holdings[ticker] = shares

            # Now simulate daily returns until next rebalance
            # Get price data for these tickers during the holding period
            holding_dates = dates[(dates >= rebal_date) & (dates < next_rebal)]
            if len(holding_dates) == 0:
                continue
            # For each day, compute portfolio value using holdings
            for day in holding_dates:
                day_prices = price_data.xs(day, level='Date')['Close']
                day_value = 0
                for ticker, shares in holdings.items():
                    if ticker in day_prices:
                        day_value += shares * day_prices[ticker]
                # If no holdings, day_value stays same
                daily_portfolio_values.append(day_value)
                # Benchmark value
                bench_price = bench.reindex([day], method='ffill').iloc[0] if not bench.empty else 1
                bench_value = self.initial_capital * (bench_price / bench.iloc[0] if not bench.empty else 1)
                daily_benchmark_values.append(bench_value)

        # Compute final metrics
        if len(daily_portfolio_values) == 0:
            return {}

        portfolio_series = pd.Series(daily_portfolio_values, index=holding_dates)
        benchmark_series = pd.Series(daily_benchmark_values, index=holding_dates)
        returns = portfolio_series.pct_change().dropna()
        bench_returns = benchmark_series.pct_change().dropna()

        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        benchmark_total_return = (benchmark_series.iloc[-1] / self.initial_capital) - 1

        excess_return = total_return - benchmark_total_return

        # Sharpe ratio (assuming risk‑free = 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Also compute hit rate (percentage of positive excess returns)
        excess_daily = returns - bench_returns
        hit_rate = (excess_daily > 0).mean()

        return {
            'strategy_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'hit_rate': hit_rate,
            'final_value': portfolio_series.iloc[-1]
        }