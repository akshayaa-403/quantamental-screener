"""
Backtesting engine – simulates historical performance of the screening strategy
with weekly rebalancing and compares against an equal‑weight benchmark.
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
from config.settings import Config


class BacktestEngine:
    """Backtests the screener strategy"""

    def __init__(self, config: Config):
        self.config = config
        self.backtest_results = {}

    def get_historical_universe(self, lookback_months: int = 12, tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get historical price data for backtesting"""
        if tickers is None:
            # For standalone use, you'd need a ticker list. In main we pass the universe.
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

        print(f"Downloading {lookback_months}-month historical data for backtesting...")
        historical_data = {}
        period = f"{lookback_months}mo"

        for i, ticker in enumerate(tickers[:30]):  # Limit to 30 for demo
            if i % 5 == 0:
                print(f"   Downloaded {i}/30")

            try:
                data = yf.download(ticker, period=period, progress=False)
                if len(data) > 100:
                    # Ensure 1D series
                    data['Close'] = data['Close'].squeeze()
                    data['Volume'] = data['Volume'].squeeze()
                    historical_data[ticker] = data
                time.sleep(0.1)
            except:
                continue

        print(f"Historical data ready for {len(historical_data)} stocks")
        return historical_data

    def run_historical_screening(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[str]:
        """Run screening logic for a historical date"""
        try:
            screened_stocks = []
            for ticker, stock_data in data.items():
                historical_subset = stock_data[stock_data.index <= date]
                if len(historical_subset) < 50:
                    continue

                # Simple momentum score for backtesting
                returns_20d = historical_subset['Close'].pct_change(20).iloc[-1]
                volume_ratio = (historical_subset['Volume'].iloc[-20:].mean() /
                               historical_subset['Volume'].iloc[-60:-20].mean())
                # Avoid Series truth value ambiguity
                if isinstance(returns_20d, pd.Series):
                    returns_20d = returns_20d.iloc[0]
                if isinstance(volume_ratio, pd.Series):
                    volume_ratio = volume_ratio.iloc[0]

                momentum_score = returns_20d * 2
                volume_score = np.log(volume_ratio) if volume_ratio > 0 else 0
                composite_score = momentum_score + volume_score * 0.3

                screened_stocks.append((ticker, composite_score))

            screened_stocks.sort(key=lambda x: x[1], reverse=True)
            return [ticker for ticker, _ in screened_stocks[:self.config.TOP_N_STOCKS]]

        except Exception as e:
            print(f"Error in historical screening for {date}: {e}")
            return []

    def calculate_portfolio_returns(self, selected_stocks: List[str],
                                   data: Dict[str, pd.DataFrame],
                                   start_date: pd.Timestamp,
                                   end_date: pd.Timestamp) -> float:
        """Calculate portfolio returns for selected stocks"""
        try:
            returns = []
            for ticker in selected_stocks:
                if ticker in data:
                    stock_data = data[ticker]
                    period_data = stock_data[(stock_data.index >= start_date) &
                                           (stock_data.index <= end_date)]
                    if len(period_data) > 1:
                        start_price = period_data['Close'].iloc[0]
                        end_price = period_data['Close'].iloc[-1]
                        # Ensure scalar
                        if isinstance(start_price, pd.Series):
                            start_price = start_price.iloc[0]
                        if isinstance(end_price, pd.Series):
                            end_price = end_price.iloc[0]
                        stock_return = (end_price / start_price) - 1
                        returns.append(stock_return)
            return np.mean(returns) if returns else 0.0

        except Exception as e:
            print(f"Error calculating portfolio returns: {e}")
            return 0.0

    def run_backtest(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run complete backtest"""
        print("Running backtest...")

        # Get date range
        start_date = min(data.index[60] for data in historical_data.values())
        end_date = max(data.index[-1] for data in historical_data.values())

        # Generate weekly screening dates
        screening_dates = pd.date_range(start=start_date, end=end_date, freq='W')

        portfolio_returns = []

        for i in range(len(screening_dates) - 1):
            current_date = screening_dates[i]
            next_date = screening_dates[i + 1]

            selected_stocks = self.run_historical_screening(historical_data, current_date)
            period_return = self.calculate_portfolio_returns(
                selected_stocks, historical_data, current_date, next_date
            )

            portfolio_returns.append({
                'date': current_date,
                'return': period_return,
                'stocks': selected_stocks
            })

            if i % 10 == 0:
                print(f"   Processed {i}/{len(screening_dates)-1} periods")

        returns_series = pd.Series([p['return'] for p in portfolio_returns])
        cumulative_returns = (1 + returns_series).cumprod()

        total_return = cumulative_returns.iloc[-1] - 1
        volatility = returns_series.std() * np.sqrt(52)
        sharpe_ratio = (returns_series.mean() * 52) / volatility if volatility > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()

        # Benchmark (equal-weight all stocks)
        all_stocks = list(historical_data.keys())
        benchmark_returns = []
        for i in range(len(screening_dates) - 1):
            current_date = screening_dates[i]
            next_date = screening_dates[i + 1]
            benchmark_return = self.calculate_portfolio_returns(
                all_stocks, historical_data, current_date, next_date
            )
            benchmark_returns.append(benchmark_return)

        benchmark_series = pd.Series(benchmark_returns)
        benchmark_cumulative = (1 + benchmark_series).cumprod()
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1

        results = {
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_total_return,
            'excess_return': total_return - benchmark_total_return,
            'cumulative_returns': cumulative_returns,
            'benchmark_cumulative': benchmark_cumulative
        }

        self.backtest_results = results
        print("Backtest completed!")
        return results

    def create_backtest_visualization(self, results: Dict) -> go.Figure:
        """Create backtest performance visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Rolling Returns',
                           'Drawdown', 'Performance Metrics'),
            specs=[[{}, {}],
                   [{}, {"type": "table"}]]
        )

        dates = [p['date'] for p in results['portfolio_returns']]

        # Cumulative returns
        fig.add_trace(go.Scatter(
            x=dates, y=results['cumulative_returns'],
            name='Strategy', line=dict(color='blue', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=results['benchmark_cumulative'],
            name='Benchmark', line=dict(color='red', width=2)
        ), row=1, col=1)

        # Rolling returns (4-week)
        rolling_returns = pd.Series([p['return'] for p in results['portfolio_returns']]).rolling(4).mean()
        fig.add_trace(go.Scatter(
            x=dates, y=rolling_returns,
            name='4-Week Rolling Returns', line=dict(color='green', width=1)
        ), row=1, col=2)

        # Drawdown
        drawdown = results['cumulative_returns'] / results['cumulative_returns'].expanding().max() - 1
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown,
            name='Drawdown', fill='tonexty',
            line=dict(color='red', width=1)
        ), row=2, col=1)

        # Performance table
        metrics_table = [
            ['Metric', 'Strategy', 'Benchmark'],
            ['Total Return', f"{results['total_return']:.2%}", f"{results['benchmark_return']:.2%}"],
            ['Volatility', f"{results['volatility']:.2%}", 'N/A'],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}", 'N/A'],
            ['Max Drawdown', f"{results['max_drawdown']:.2%}", 'N/A'],
            ['Excess Return', f"{results['excess_return']:.2%}", 'N/A']
        ]
        fig.add_trace(go.Table(
            header=dict(values=metrics_table[0]),
            cells=dict(values=list(zip(*metrics_table[1:])))
        ), row=2, col=2)

        fig.update_layout(
            title='Backtest Results Dashboard',
            template='plotly_white',
            height=800
        )
        return fig