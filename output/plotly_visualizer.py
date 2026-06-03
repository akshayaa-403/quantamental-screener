import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class PlotlyVisualizer:
    @staticmethod
    def create_score_distribution(scores_df: pd.DataFrame):
        fig = px.histogram(scores_df, x='composite', title='Composite Score Distribution')
        return fig
    
    @staticmethod
    def create_individual_chart(ticker: str, price_data: pd.DataFrame):
        ticker_data = price_data.xs(ticker, level='Ticker')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], mode='lines', name='Close'))
        return fig