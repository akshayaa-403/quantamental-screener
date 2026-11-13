import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config.settings import get_settings
import logging
import yfinance as yf
from utils.logging_utils import retry_on_network_error

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_TICKER = "^IRX"  # 13-week US T-bill yield, in percent


class BacktestEngine:
    def __init__(self, config: dict = None):
        settings = get_settings()
        config = config or {}
        self.period = config.get('period', settings.backtest.period)
        self.rebalance_freq = config.get('rebalance_freq', settings.backtest.rebalance_freq)
        self.top_n = config.get('top_n', settings.backtest.top_n_stocks)
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.benchmark_ticker = config.get('benchmark', 'SPY')
        # Annualized risk-free rate for the Sharpe ratio. None = daily T-bill yield.
        self.risk_free_rate = config.get('risk_free_rate', settings.backtest.risk_free_rate)
        # Cost charged on the notional of every buy and sell, in basis points.
        self.cost_rate = config.get('transaction_cost_bps', settings.backtest.transaction_cost_bps) / 1e4
        # Point-in-time liquidity screen: trailing 20-day average dollar volume,
        # known as of the previous close. 0 disables it.
        self.min_dollar_volume = config.get('min_dollar_volume', settings.universe.min_dollar_volume)
        self.n_bootstrap = config.get('n_bootstrap', settings.backtest.n_bootstrap)
        self.holdout_fraction = config.get('holdout_fraction', settings.backtest.holdout_fraction)
        self.seed = config.get('seed', 0)

    # --- market data ---------------------------------------------------------
    @staticmethod
    def _download_close(ticker: str, start_date, end_date) -> pd.Series:
        # yfinance's end is exclusive; without the extra day the series misses
        # the backtest's last session. auto_adjust matches the strategy's
        # dividend-adjusted closes.
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        frame = yf.download(ticker, start=start_date, end=end, auto_adjust=True, progress=False)
        if frame.empty:
            logger.warning(f"No data for {ticker}")
            return pd.Series(dtype=float)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        series = frame['Close']
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        return series

    @retry_on_network_error
    def _fetch_benchmark(self, start_date, end_date) -> pd.Series:
        return self._download_close(self.benchmark_ticker, start_date, end_date)

    @retry_on_network_error
    def _fetch_risk_free(self, start_date, end_date) -> pd.Series:
        """Daily 13-week T-bill yield as an annual decimal (4.3% -> 0.043)."""
        return self._download_close(RISK_FREE_TICKER, start_date, end_date) / 100.0

    def _risk_free_daily(self, dates: pd.DatetimeIndex) -> tuple:
        """(daily risk-free return per date, annualized mean rate, source label)."""
        if self.risk_free_rate is not None:
            rate = float(self.risk_free_rate)
            return pd.Series(rate / TRADING_DAYS, index=dates), rate, "fixed"
        try:
            annual = self._fetch_risk_free(dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'))
        except Exception as e:  # noqa: BLE001 - the Sharpe ratio degrades, the backtest doesn't fail
            logger.warning(f"Risk-free rate download failed: {e}")
            annual = pd.Series(dtype=float)
        annual = annual.reindex(dates).ffill().bfill()
        if annual.isna().all():
            logger.warning("No T-bill data; the Sharpe ratio uses a 0% risk-free rate and is overstated.")
            return pd.Series(0.0, index=dates), 0.0, "unavailable (0%)"
        # A T-bill yield is quoted per year; one trading day earns 1/252 of it.
        return annual / TRADING_DAYS, float(annual.mean()), "13-week T-bill"

    # --- simulation ----------------------------------------------------------
    def _rebalance_to_equal_weight(self, holdings: Dict[str, float], cash: float,
                                   target: List[str], prices: pd.Series) -> tuple:
        """
        Trade to an equal weight in each name of ``target`` and nothing else.

        Costs are paid out of the portfolio, so the per-name target T solves
        ``N*T + c * sum|T - current_i| = V`` (V = value before trading). The
        cost term is tiny, so a few fixed-point steps converge to float
        precision. Returns (holdings, cash, traded notional, costs).
        """
        current = {t: shares * prices[t] for t, shares in holdings.items()}
        value = cash + sum(current.values())
        if not target:
            sold = sum(current.values())
            cost = sold * self.cost_rate
            return {}, value - cost, sold, cost

        n = len(target)
        exits = sum(v for t, v in current.items() if t not in target)
        per_name = value / n
        for _ in range(20):
            traded = exits + sum(abs(per_name - current.get(t, 0.0)) for t in target)
            new = (value - self.cost_rate * traded) / n
            if abs(new - per_name) < 1e-9:
                per_name = new
                break
            per_name = new
        traded = exits + sum(abs(per_name - current.get(t, 0.0)) for t in target)
        cost = self.cost_rate * traded
        new_holdings = {t: per_name / prices[t] for t in target}
        return new_holdings, value - n * per_name - cost, traded, cost

    def run(self, scoring_engine, price_data: pd.DataFrame, sentiment_scores: Optional[Dict] = None,
            start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Simulate an equal-weight portfolio of the scorer's top ``top_n`` names.

        * Signals are lagged one day: the portfolio chosen on day t uses scores
          computed from data up to the close of t-1, and trades at t's close.
        * Only point-in-time factors are used (``composite_pit``): sentiment is
          a snapshot of today's news, and copying it onto past dates is
          lookahead. ``sentiment_scores`` is accepted and ignored.
        * Every rebalance resets the portfolio to equal weights (1/N each), so
          the strategy's per-period return is exactly what the significance
          test compares against random portfolios.
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

        scores_df = scoring_engine.compute(price_data, None)
        if scores_df.empty:
            logger.warning("Scores DataFrame is empty.")
            return {}

        # Trade on yesterday's signal.
        signal = scores_df['composite_pit'].groupby(level='Ticker').shift(1).unstack('Ticker')
        raw_close = price_data['Close'].unstack('Ticker')
        close = raw_close.ffill()  # value a position through a missing print
        signal = signal.reindex(index=close.index, columns=close.columns)

        eligible = signal.notna() & raw_close.notna()
        if self.min_dollar_volume and 'Volume' in price_data.columns:
            dollar_volume = (price_data['Close'] * price_data['Volume']).unstack('Ticker').reindex_like(close)
            eligible &= dollar_volume.rolling(20).mean().shift(1) >= self.min_dollar_volume

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
                target = list(pool.nlargest(self.top_n).index)
                picks[current_date] = (target, pool.index)
                holdings, cash, day_traded, day_cost = self._rebalance_to_equal_weight(
                    holdings, cash, target, prices)
                traded += day_traded
                costs += day_cost

            total_value = cash + sum(shares * prices[t] for t, shares in holdings.items())
            portfolio_values.append(total_value)

        if len(portfolio_values) < 2:
            logger.warning("Backtest generated too few values.")
            return {}

        port_series = pd.Series(portfolio_values, index=dates)
        universe_series = self._equal_weight_universe(picks, close, dates)

        bench = self._fetch_benchmark(dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'))
        bench = bench.reindex(dates).ffill().bfill()
        if bench.isna().all():
            logger.warning("Benchmark data entirely missing. Using flat 0% return.")
            bench = pd.Series(1.0, index=dates)
        bench_series = self.initial_capital * bench / bench.iloc[0]

        rf_daily, rf_annual, rf_source = self._risk_free_daily(dates)
        port_returns = port_series.pct_change().dropna()
        bench_returns = bench_series.pct_change().dropna()

        total_return = (port_series.iloc[-1] / self.initial_capital) - 1
        benchmark_total_return = (bench_series.iloc[-1] / self.initial_capital) - 1
        universe_total_return = (universe_series.iloc[-1] / self.initial_capital) - 1

        # Sharpe = mean(r - rf) / std(r - rf) * sqrt(252), with rf the daily
        # risk-free return on the same day.
        excess_rf = port_returns - rf_daily.reindex(port_returns.index)
        sharpe = (excess_rf.mean() / excess_rf.std() * np.sqrt(TRADING_DAYS)
                  if excess_rf.std() > 0 else 0.0)

        cumulative = (1 + port_returns).cumprod()
        drawdown = cumulative / cumulative.expanding().max() - 1
        max_drawdown = drawdown.min()
        volatility = port_returns.std() * np.sqrt(TRADING_DAYS)

        excess_daily = port_returns - bench_returns.reindex(port_returns.index).ffill()
        hit_rate = (excess_daily > 0).mean()

        p_value, n_periods = self._selection_p_value(picks, close, dates[-1])

        return {
            'strategy_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': total_return - benchmark_total_return,
            # Equal-weight buy of every eligible name: the fair yardstick for
            # stock selection, since it shares the universe's biases (today's
            # index members, ranked by today's market cap) with the strategy.
            'universe_return': universe_total_return,
            'excess_vs_universe': total_return - universe_total_return,
            'sharpe_ratio': sharpe,
            'risk_free_rate': rf_annual,
            'risk_free_source': rf_source,
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
            'holdout': self._holdout(picks, close, dates, port_series, universe_series, bench_series),
            'portfolio_series': port_series,
            'benchmark_series': bench_series,
            'universe_series': universe_series,
        }

    def _equal_weight_universe(self, picks: Dict, close: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        """Equal weight in every eligible name, reset at each rebalance, gross of costs."""
        values = pd.Series(np.nan, index=dates)
        value = float(self.initial_capital)
        starts = sorted(picks)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else dates[-1]
            _, pool = picks[start]
            span = dates[(dates >= start) & (dates <= end)]
            if len(pool):
                growth = (close.loc[span, pool] / close.loc[start, pool]).mean(axis=1)
            else:
                growth = pd.Series(1.0, index=span)
            path = value * growth
            # The end date belongs to the next period (rebalanced at its close),
            # except on the last period.
            own = span if i + 1 == len(starts) else span[span < end]
            values.loc[own] = path.loc[own].to_numpy()
            value = float(path.iloc[-1])
        return values.ffill()

    def _holdout(self, picks, close, dates, port, universe, bench) -> Dict:
        """Metrics on the last ``holdout_fraction`` of the window only."""
        starts = sorted(picks)
        if not self.holdout_fraction or len(starts) < 2:
            return {}
        cut = dates[int(len(dates) * (1 - self.holdout_fraction))]
        hold = [d for d in starts if d >= cut]
        if not hold:
            return {}
        first = hold[0]

        def ret(series):
            return float(series.iloc[-1] / series.loc[first] - 1)

        p_value, n = self._selection_p_value({d: picks[d] for d in hold}, close, dates[-1])
        return {
            'start_date': first,
            'strategy_return': ret(port),
            'universe_return': ret(universe),
            'benchmark_return': ret(bench),
            'p_value': p_value,
            'n_rebalances': n,
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
            if start == end or not len(pool) or not len(target):
                continue
            period = (close.loc[end, pool] / close.loc[start, pool] - 1).fillna(0.0)
            k = len(target)
            strat_log += np.log1p(period[list(target)].mean())
            draws = rng.random((self.n_bootstrap, len(pool))).argsort(axis=1)[:, :k]
            rand_log += np.log1p(period.to_numpy()[draws].mean(axis=1))
        p_value = (1 + np.sum(rand_log >= strat_log)) / (self.n_bootstrap + 1)
        return float(p_value), len(starts)
