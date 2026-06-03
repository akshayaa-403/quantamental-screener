from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class OutputFormatter(ABC):
    @abstractmethod
    def display_results(self,
                        scores_df: pd.DataFrame,
                        backtest_results: Dict,
                        sector_dist: pd.Series,
                        factor_corr: pd.DataFrame):
        pass