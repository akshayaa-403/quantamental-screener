"""
Visualization module – creates interactive Plotly dashboards for score distributions,
sector breakdowns, factor correlations, top stocks, and detailed price charts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from config.settings import Config


class Visualizer:
    """Creates visualizations and dashboards"""

    def __init__(self, config: Config, price_data: dict):
        self.config = config
        self.price_data = price_data

    def create_score_distribution(self, scores_df: pd.DataFrame) -> go.Figure:
        """Create score distribution histogram"""
        # Force numeric and ensure 1D
        scores = pd.to_numeric(scores_df['composite_score'], errors='coerce').dropna().values
        if len(scores) == 0:
            scores = np.array([0.0])
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name='Score Distribution',
            marker_color='rgba(55, 83, 109, 0.7)',
            marker_line_color='rgba(55, 83, 109, 1.0)',
            marker_line_width=2
        ))
        fig.update_layout(
            title='Composite Score Distribution',
            xaxis_title='Composite Score',
            yaxis_title='Number of Stocks',
            template='plotly_white'
        )
        return fig

    def create_sector_breakdown(self, scores_df: pd.DataFrame) -> go.Figure:
        """Create sector breakdown pie chart"""
        sector_counts = scores_df['sector'].value_counts()
        if len(sector_counts) == 0:
            return go.Figure()
        fig = go.Figure(data=[go.Pie(
            labels=sector_counts.index.tolist(),
            values=sector_counts.values.tolist(),
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )])
        fig.update_layout(
            title='Sector Distribution in Universe',
            template='plotly_white'
        )
        return fig

    def create_factor_correlation(self, scores_df: pd.DataFrame) -> go.Figure:
        """Create factor correlation heatmap"""
        factor_cols = ['momentum_score', 'sentiment_score', 'volume_score', 'volatility_score']
        # Ensure numeric
        corr_df = scores_df[factor_cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = corr_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        fig.update_layout(
            title='Factor Correlation Matrix',
            template='plotly_white'
        )
        return fig

    def create_top_stocks_chart(self, top_stocks: pd.DataFrame) -> go.Figure:
        """Create horizontal bar chart of top stocks"""
        # Convert to numeric and ensure 1D
        scores = pd.to_numeric(top_stocks['composite_score'], errors='coerce').dropna().values
        tickers = top_stocks['ticker'].astype(str).values
        if len(scores) == 0:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tickers,
            x=scores,
            orientation='h',
            marker_color=scores,
            marker_colorscale='Viridis',
            text=[f"{x:.3f}" for x in scores],
            textposition='auto'
        ))
        fig.update_layout(
            title=f'Top {len(top_stocks)} Stocks by Composite Score',
            xaxis_title='Composite Score',
            yaxis_title='Stock Ticker',
            template='plotly_white',
            height=400 + len(top_stocks) * 20
        )
        return fig

    def create_factor_breakdown(self, top_stocks: pd.DataFrame) -> go.Figure:
        """Create stacked bar chart showing factor contributions"""
        tickers = top_stocks['ticker'].astype(str).values
        # Convert each factor to numeric, multiply by weight
        momentum = pd.to_numeric(top_stocks['momentum_score'], errors='coerce') * self.config.MOMENTUM_WEIGHT
        sentiment = pd.to_numeric(top_stocks['sentiment_score'], errors='coerce') * self.config.SENTIMENT_WEIGHT
        volume = pd.to_numeric(top_stocks['volume_score'], errors='coerce') * self.config.VOLUME_WEIGHT
        volatility = pd.to_numeric(top_stocks['volatility_score'], errors='coerce') * self.config.VOLATILITY_WEIGHT

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Momentum', x=tickers, y=momentum, marker_color='blue'))
        fig.add_trace(go.Bar(name='Sentiment', x=tickers, y=sentiment, marker_color='green'))
        fig.add_trace(go.Bar(name='Volume', x=tickers, y=volume, marker_color='orange'))
        fig.add_trace(go.Bar(name='Volatility', x=tickers, y=volatility, marker_color='red'))
        fig.update_layout(
            title='Factor Contribution Breakdown',
            xaxis_title='Stock Ticker',
            yaxis_title='Weighted Factor Score',
            barmode='stack',
            template='plotly_white'
        )
        return fig

    def create_price_chart(self, ticker: str) -> go.Figure:
        """Create detailed price chart with indicators"""
        if ticker not in self.price_data:
            return go.Figure().add_annotation(text=f"No data for {ticker}")

        data = self.price_data[ticker].copy()
        # Ensure series are 1D
        close = data['Close'].squeeze()
        open_ = data['Open'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()
        volume = data['Volume'].squeeze()

        # Calculate indicators
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        bb = ta.volatility.BollingerBands(close)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['RSI'] = ta.momentum.rsi(close)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{ticker} Price Chart', 'Volume', 'RSI'),
            row_width=[0.7, 0.15, 0.15]
        )

        # Price
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=open_,
            high=high,
            low=low,
            close=close,
            name='Price'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            line=dict(color='blue', width=1),
            name='SMA 20'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            line=dict(color='red', width=1),
            name='SMA 50'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            line=dict(color='gray', width=1, dash='dash'),
            name='BB Upper'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            line=dict(color='gray', width=1, dash='dash'),
            name='BB Lower'
        ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=data.index, y=volume,
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            line=dict(color='purple', width=2),
            name='RSI'
        ), row=3, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            template='plotly_white',
            height=800,
            showlegend=True
        )
        fig.update_xaxes(rangeslider_visible=False)
        return fig

    def create_dashboard(self, scores_df: pd.DataFrame, top_stocks: pd.DataFrame):
        """Create comprehensive dashboard"""
        try:
            print("Creating visualization dashboard...")
            # Show each plot individually so errors are isolated
            fig1 = self.create_score_distribution(scores_df)
            fig1.show()
            fig2 = self.create_sector_breakdown(scores_df)
            fig2.show()
            fig3 = self.create_factor_correlation(scores_df)
            fig3.show()
            fig4 = self.create_top_stocks_chart(top_stocks)
            fig4.show()
            fig5 = self.create_factor_breakdown(top_stocks)
            fig5.show()
            print("Creating individual stock charts...")
            for ticker in top_stocks['ticker'].head(3):
                price_fig = self.create_price_chart(ticker)
                price_fig.show()
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")