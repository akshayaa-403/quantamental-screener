import pandas as pd
from typing import Dict


def _fmt(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, float):
        return f"{value:,.4f}"
    return value


class ConsoleReporter:
    def display_results(self, scores_df: pd.DataFrame, backtest_results: Dict, top_n: int = 10):
        print("\n" + "="*80)
        print("QUANTAMENTAL EQUITY SCREENER - FINAL REPORT")
        print("="*80)
        print(f"\nTotal stocks analyzed: {len(scores_df)}")
        print(f"Top {top_n} recommendations:")
        print("(rank_pit = rank on the price factors only: the ranking the backtest tested)")
        top = scores_df.nlargest(top_n, 'composite')[['composite'] + [c for c in scores_df.columns if c != 'composite']]
        print(top.to_string())
        print("\nBacktest (price factors only; equal weight, trades on the previous day's signal).")
        print("Judge selection skill by p_value and excess_vs_universe; the universe is today's")
        print("largest S&P 500 names, so beating SPY (excess_return) is flattered by hindsight.")
        for key, value in backtest_results.items():
            if isinstance(value, (pd.Series, pd.DataFrame, dict)):
                continue
            print(f"  {key:<20} {_fmt(value)}")
        holdout = backtest_results.get('holdout') or {}
        if holdout:
            print("\nHoldout (last part of the window; check tuned weights here):")
            for key, value in holdout.items():
                print(f"  {key:<20} {_fmt(value)}")
        print("="*80)
