from core.output_formatter import OutputFormatter
import pandas as pd
from typing import Dict

class ConsoleReporter(OutputFormatter):
    def display_results(self, scores_df: pd.DataFrame, backtest_results: Dict,
                        sector_dist: pd.Series, factor_corr: pd.DataFrame):
        print("\n" + "="*80)
        print("QUANTAMENTAL EQUITY SCREENER - FINAL REPORT")
        print("="*80)
        print(f"\nTotal stocks analyzed: {len(scores_df)}")
        print(f"Top 10 recommendations:")
        top10 = scores_df.nlargest(10, 'composite')[['composite'] + [c for c in scores_df.columns if c != 'composite']]
        print(top10.to_string())
        print(f"\nBacktest Results: {backtest_results}")
        print("="*80)