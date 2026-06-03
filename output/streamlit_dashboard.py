import streamlit as st
import plotly.express as px
import pandas as pd
from core.output_formatter import OutputFormatter
from typing import Dict

class StreamlitDashboard(OutputFormatter):
    def display_results(self, scores_df: pd.DataFrame, backtest_results: Dict,
                        sector_dist: pd.Series, factor_corr: pd.DataFrame):
        st.title("Quantamental Screener Dashboard")
        st.subheader("Top Stocks")
        st.dataframe(scores_df.head(10))
        st.subheader("Sector Distribution")
        fig = px.pie(values=sector_dist.values, names=sector_dist.index)
        st.plotly_chart(fig)
        st.subheader("Factor Correlation Heatmap")
        fig2 = px.imshow(factor_corr, text_auto=True)
        st.plotly_chart(fig2)
        st.subheader("Backtest Metrics")
        st.json(backtest_results)