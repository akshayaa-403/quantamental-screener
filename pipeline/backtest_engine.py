import pandas as pd
import numpy as np
from typing import Dict, Optional
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
        # Cost charged on the notional of every buy and sell, in basis points.
        self.cost_rate = config.get('transaction_cost_bps', settings.backtest.transaction_cost_bps) / 1e4
        # Point-in-time liquidity screen: trailing 20-day average share volume,
        # known as of the previous close. 0 disables it.
        self.min_volume = config.get('min_volume', settings.universe.min_volume)
        self.n_bootstrap = config.get('n_bootstrap', settings.backtest.n_bootstrap)
        self.seed = config.get('seed', 0)

    @retry_on_network_error
    def _fetch_benchmark(self, start_date, end_date) -> pd.Series:
        # yfinance's end is exclusive; without the extra day the benchmark
        # misses the backtest's last session. auto_adjust matches the
        # strategy's dividend-adjusted closes.
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        bench = yf.download(self.benchmark_ticker, start=start_date, end=end,
                            auto_adjust=True, progress=False)
        if bench.empty:
            logger.warning(f"No benchmark data for {self.benchmark_ticker}")
            return pd.Series(dtype=float)

        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
        price_series = bench['Close']
        if price_series.index.tz is not None:
            price_series.index = price_series.index.tz_localize(None)
        return price_series

    def run(self, scoring_engine, price_data: pd.DataFrame, sentiment_scores: Optional[Dict] = None,
            start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Simulate trading the scorer's top ``top_n`` names.

        * Signals are lagged one day: the portfolio chosen on day t uses scores
          computed from data up to the close of t-1, and trades at t's close.
        * Sentiment is never used. ``sentiment_scores`` is accepted for
          backwards compatibility and ignored: the scores are a snapshot of
          today's news, and copying them onto past dates is lookahead.
        * Trading starts on the first day enough names are out of warm-up.
          ``start_date`` / ``end_date`` restrict the trading window while still
          using earlier data for warm-up, e.g. for an out-of-sample period.
        """
        logger.info("Starting backtest engine...")
        if price_data.empty:
            logger.error("No price data provided.")
            return {}
        if sentiment_scores:
            logger.warning("Backtest ignores sentiment_scores: there is no point-in-time sentiment history.")

        price_data = price_data.sort_index()
        if price_data.index.get_level_values('Date').tz is not None:
            price_data = price_data.reset_index()
            price_data['Date'] = price_data['Date'].dt.tz_localize(None)
            price_data = price_data.set_index(['Ticker', 'Date']).sort_index()

        scores_df = scoring_engine.compute(price_data, None, weights={'sentiment': 0.0})
        if scores_df.empty:
            logger.warning("Scores DataFrame is empty.")
            return {}

        # Trade on yesterday's signal.
        signal = scores_df['composite'].groupby(level='Ticker').shift(1).unstack('Ticker')
        raw_close = price_data['Close'].unstack('Ticker')
        close = raw_close.ffill()  # value a position through a missing print
        signal = signal.reindex(index=close.index, columns=close.columns)

        eligible = signal.notna() & raw_close.notna()
        if self.min_volume and 'Volume' in price_data.columns:
            adv = price_data['Volume'].unstack('Ticker').reindex_like(close)
            eligible &= adv.rolling(20).mean().shift(1) >= self.min_volume

        dates = close.index
        in_window = np.ones(len(dates), dtype=bool)
        if start_date:
            in_window &= dates >= pd.Timestamp(start_date)
        if end_date:
            in_window &= dates <= pd.Timestamp(end_date)
        window = dates[in_window]
        ready = eligible.loc[window].sum(axis=1) >= self.top_n
        if not ready.any():
            logger.warning("No date in the window has enough scored names; history too short for warm-up.")
            return {}
        dates = window[window >= ready.idxmax()]
        rebalance_dates = set(dates[::self.rebalance_freq])
        logger.info(f"Backtest {dates[0].date()} -> {dates[-1].date()}: {len(rebalance_dates)} rebalances, "
                    f"top {self.top_n}, {self.cost_rate * 1e4:.1f} bps per trade")

        cash = float(self.initial_capital)
        holdings: Dict[str, float] = {}
        picks = {}
        traded = 0.0
        costs = 0.0
        portfolio_values = []

        for current_date in dates:
            prices = close.loc[current_date]
            if current_date in rebalance_dates:
                pool = signal.loc[current_date][eligible.loc[current_date]]
                target = set(pool.nlargest(self.top_n).index)
                picks[current_date] = (target, pool.index)

                # Trade only the names that changed; continuing positions stay.
                sells = [t for t in holdings if t not in target]
                for ticker in sells:
                    notional = holdings.pop(ticker) * prices[ticker]
                    cash += notional * (1 - self.cost_rate)
                    traded += notional
                    costs += notional * self.cost_rate
                buys = [t for t in target if t not in holdings]
                if buys:
                    per_stock = cash / len(buys)
                    for ticker in buys:
                        holdings[ticker] = per_stock * (1 - self.cost_rate) / prices[ticker]
                        traded += per_stock
                        costs += per_stock * self.cost_rate
                    cash = 0.0
                logger.debug(f"Rebalance {current_date.date()}: sold {len(sells)}, bought {len(buys)}")

            total_value = cash + sum(shares * prices[t] for t, shares in holdings.items())
            portfolio_values.append(total_value)
            logger.debug(f"{current_date.date()}: value={total_value:,.2f}")

        if len(portfolio_values) < 2:
            logger.warning("Backtest generated too few values.")
            return {}

        port_series = pd.Series(portfolio_values, index=dates)
        bench = self._fetch_benchmark(dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'))
        bench = bench.reindex(dates).ffill().bfill()
        if bench.isna().all():
            logger.warning("Benchmark data entirely missing. Using flat 0% return.")
            bench = pd.Series(1.0, index=dates)
        bench_series = self.initial_capital * bench / bench.iloc[0]

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

        p_value, n_periods = self._selection_p_value(picks, close, dates[-1])

        return {
            'strategy_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'hit_rate': hit_rate,
            'p_value': p_value,
            'n_rebalances': n_periods,
            'turnover': traded / self.initial_capital,
            'total_costs': costs,
            'start_date': dates[0],
            'end_date': dates[-1],
            'final_value': port_series.iloc[-1],
            'initial_capital': self.initial_capital,
            'portfolio_series': port_series,
            'benchmark_series': bench_series,
        }

    def _selection_p_value(self, picks: Dict, close: pd.DataFrame, last_date) -> tuple:
        """
        Random-portfolio test of the stock selection.

        For each holding period, draw ``n_bootstrap`` random portfolios of the
        same size from the same eligible pool the strategy chose from, and
        compare the compounded equal-weight return of the strategy's picks with
        that of the random picks. The p-value is the share of random portfolios
        that did at least as well (one-sided, gross of costs, with the +1
        correction so it is never exactly 0). Few periods means little
        evidence: read a p-value from ~25 rebalances with care.
        """
        starts = sorted(picks)
        if not starts or self.n_bootstrap <= 0:
            return np.nan, len(starts)
        rng = np.random.default_rng(self.seed)
        ends = starts[1:] + [last_date]
        strat_log = 0.0
        rand_log = np.zeros(self.n_bootstrap)
        for start, end in zip(starts, ends):
            target, pool = picks[start]
            if start == end or not len(pool):
                continue
            period = (close.loc[end, pool] / close.loc[start, pool] - 1).fillna(0.0)
            k = len(target)
            strat_log += np.log1p(period[list(target)].mean())
            draws = rng.random((self.n_bootstrap, len(pool))).argsort(axis=1)[:, :k]
            rand_log += np.log1p(period.to_numpy()[draws].mean(axis=1))
        p_value = (1 + np.sum(rand_log >= strat_log)) / (self.n_bootstrap + 1)
        return float(p_value), len(starts)
