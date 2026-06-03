from core.output_formatter import OutputFormatter
import pandas as pd
from typing import Dict

class CSVExporter(OutputFormatter):
    def display_results(self, scores_df: pd.DataFrame, backtest_results: Dict,
                        sector_dist: pd.Series, factor_corr: pd.DataFrame):
        scores_df.to_csv("top_stocks_recommendations.csv", index=True)
        print("Saved results to top_stocks_recommendations.csv")