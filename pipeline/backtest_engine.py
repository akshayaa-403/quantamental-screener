import pandas as pd
import numpy as np
from typing import Dict, Optional
from config.settings import get_settings
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.period = config.get('period', settings.backtest.period)
        self.rebalance_freq = config.get('rebalance_freq', settings.backtest.rebalance_freq)  # trading days
        self.top_n = config.get('top_n', settings.backtest.top_n_stocks)
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.benchmark_ticker = config.get('benchmark', 'SPY')

    def run(self, scoring_engine, price_data: pd.DataFrame, sentiment_scores: Optional[Dict] = None) -> Dict:
        """
        Run backtest with historical rebalancing.
        - price_data: MultiIndex (Ticker, Date) with columns Open, High, Low, Close, Volume
        - sentiment_scores: dict of ticker -> sentiment dict (optional; if None, sentiment weight is zeroed)
        """
        price_data = price_data.sort_index()
        # If no sentiment, zero out sentiment factor weight
        if sentiment_scores is None:
            for factor in scoring_engine.factors:
                if factor.name == 'sentiment':
                    factor.weight = 0.0
                    logger.info("Sentiment factor weight set to 0 for backtest (no historical sentiment data)")

        # Compute composite scores per (Date, Ticker)
        scores_df = scoring_engine.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
        # scores_df has MultiIndex (Date, Ticker) and 'composite' column

        # Get all unique dates sorted
        dates = price_data.index.get_level_values('Date').unique().sort_values()
        if len(dates) < 2:
            logger.warning("Not enough dates for backtest.")
            return {}

        # Prepare benchmark data (SPY) for the same period
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')
        bench = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
        if bench.empty:
            logger.warning(f"Could not fetch benchmark {self.benchmark_ticker}. Using 0% return.")
            bench_prices = pd.Series(1.0, index=dates)
        else:
            bench = bench['Adj Close'].reindex(dates, method='ffill')
            # Fill any missing with last available
            bench = bench.fillna(method='ffill').fillna(method='bfill')
            bench_prices = bench

        # Define rebalance dates: use trading day indices
        rebalance_indices = list(range(0, len(dates), self.rebalance_freq))
        if not rebalance_indices:
            rebalance_indices = [0]

        # Portfolio state
        cash = self.initial_capital
        holdings = {}  # ticker -> shares
        portfolio_values = []
        benchmark_values = []

        # We'll iterate through each date in order
        for idx, current_date in enumerate(dates):
            # Check if today is a rebalance date
            is_rebalance = (idx in rebalance_indices)

            # Get prices for all tickers on this date
            try:
                day_prices = price_data.xs(current_date, level='Date')['Close']
            except KeyError:
                # No data for this date; keep previous holdings and value unchanged
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])
                    benchmark_values.append(benchmark_values[-1])
                else:
                    portfolio_values.append(cash)
                    benchmark_values.append(cash)
                continue

            # If rebalance, select top N stocks based on composite score at this date
            if is_rebalance:
                try:
                    date_scores = scores_df.xs(current_date, level='Date')
                except KeyError:
                    # No scores for this date; skip rebalance (keep holdings)
                    pass
                else:
                    date_scores = date_scores.dropna(subset=['composite'])
                    if not date_scores.empty:
                        # Select top N (if fewer, take all)
                        top_tickers = date_scores.nlargest(self.top_n, 'composite').index.tolist()
                        # Get available prices for these tickers
                        prices_for_selection = day_prices.reindex(top_tickers).dropna()
                        if not prices_for_selection.empty:
                            # Sell all current holdings at today's prices
                            if holdings:
                                # Compute cash from selling
                                sell_value = 0.0
                                for ticker, shares in holdings.items():
                                    if ticker in day_prices:
                                        sell_value += shares * day_prices[ticker]
                                cash = sell_value  # assume we sell everything
                                holdings = {}
                            # Buy new portfolio equally weighted
                            invest_per_stock = cash / len(prices_for_selection)
                            for ticker, price in prices_for_selection.items():
                                shares = invest_per_stock / price
                                holdings[ticker] = shares
                            # cash becomes 0 (fully invested)
                            cash = 0.0

            # Compute current portfolio value based on holdings and today's prices
            total_value = cash
            for ticker, shares in holdings.items():
                if ticker in day_prices:
                    total_value += shares * day_prices[ticker]
                # else: if ticker missing, value = 0 (delisted)
            portfolio_values.append(total_value)

            # Benchmark value
            bench_price = bench_prices.loc[current_date]
            bench_value = self.initial_capital * (bench_price / bench_prices.iloc[0])
            benchmark_values.append(bench_value)

        if len(portfolio_values) < 2:
            return {}

        # Convert to Series
        port_series = pd.Series(portfolio_values, index=dates)
        bench_series = pd.Series(benchmark_values, index=dates)

        # Compute returns
        port_returns = port_series.pct_change().dropna()
        bench_returns = bench_series.pct_change().dropna()

        # Metrics
        total_return = (port_series.iloc[-1] / self.initial_capital) - 1
        benchmark_total_return = (bench_series.iloc[-1] / self.initial_capital) - 1
        excess_return = total_return - benchmark_total_return
        sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252) if port_returns.std() != 0 else 0
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        volatility = port_returns.std() * np.sqrt(252)
        # Hit rate (excess returns positive)
        excess_daily = port_returns - bench_returns
        hit_rate = (excess_daily > 0).mean()

        return {
            'strategy_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'hit_rate': hit_rate,
            'final_value': port_series.iloc[-1],
            'portfolio_series': port_series,  # for plotting
            'benchmark_series': bench_series,
        }