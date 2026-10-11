import pandas as pd
from typing import Dict

class ConsoleReporter:
    def display_results(self, scores_df: pd.DataFrame, backtest_results: Dict, top_n: int = 10):
        print("\n" + "="*80)
        print("QUANTAMENTAL EQUITY SCREENER - FINAL REPORT")
        print("="*80)
        print(f"\nTotal stocks analyzed: {len(scores_df)}")
        print(f"Top {top_n} recommendations:")
        top = scores_df.nlargest(top_n, 'composite')[['composite'] + [c for c in scores_df.columns if c != 'composite']]
        print(top.to_string())
        print("\nBacktest (no sentiment; trades on the previous day's signal):")
        for key, value in backtest_results.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                continue
            if isinstance(value, float):
                value = f"{value:,.4f}"
            print(f"  {key:<18} {value}")
        print("="*80)
