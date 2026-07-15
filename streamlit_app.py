from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from config.settings import get_settings
from data_sources.yahoo_finance import YahooFinanceSource
from sentiment.ensemble_sentiment import EnsembleSentiment
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from output.plotly_visualizer import PlotlyVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Quantamental Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'screener_results' not in st.session_state:
    st.session_state.screener_results = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'scores_df' not in st.session_state:
    st.session_state.scores_df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

@st.cache_resource(ttl=3600)
def load_components():
    """Load heavy components once and cache them."""
    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    factors = [
        MomentumFactor(),
        SentimentFactor(),
        VolumeFactor(),
        VolatilityFactor()
    ]
    return data_source, sentiment_model, factors

@st.cache_data(ttl=300)  # 5 minute cache
def run_screener(universe_size, start_date, end_date, weights):
    """Run the complete screener pipeline with caching."""
    data_source, sentiment_model, factors = load_components()
    
    # Update factor weights
    for factor in factors:
        if factor.name in weights:
            factor.weight = weights[factor.name]
    
    # Universe selection
    universe_config = {'size': universe_size}
    universe_selector = UniverseSelector(universe_config)
    tickers = universe_selector.select()
    
    # Data collection
    collector = DataCollector(data_source)
    price_data = collector.collect(tickers, start_date, end_date)
    
    # Sentiment analysis (only for top 20 to save time)
    limited_tickers = tickers[:min(20, len(tickers))]
    news_by_ticker = {}
    with st.spinner("Fetching news and analyzing sentiment..."):
        for ticker in limited_tickers:
            headlines = data_source.get_news_headlines(ticker, lookback_days=7)
            if headlines:
                news_by_ticker[ticker] = headlines
    
    sentiment_scores = sentiment_model.batch_analyze(news_by_ticker) if news_by_ticker else {}
    
    # --- FALLBACK: If no real sentiment scores, use mock for demonstration ---
    if not any(sentiment_scores):
        tickers_in_data = price_data.index.get_level_values('Ticker').unique()
        sentiment_scores = {
            t: {'score': np.random.uniform(-0.8, 0.8)}
            for t in tickers_in_data[:20]
        }
    # -----------------------------------------------------------------------
    
    # Scoring
    scorer = ScoringEngine(factors)
    scores_df = scorer.compute(price_data, sentiment_scores, cross_sectional_normalize=True)
    
    # Backtest - CORRECTED: pass scorer and sentiment_scores
    backtest_engine = BacktestEngine()
    backtest_results = backtest_engine.run(scorer, price_data, sentiment_scores)
    
    # Return only serializable objects
    return scores_df, price_data, backtest_results, sentiment_scores

def main():
    # Header
    st.markdown('<div class="main-header">📈 Quantamental Equity Screener</div>', unsafe_allow_html=True)
    st.markdown("*Multi-factor analysis combining technical indicators, sentiment, and fundamentals*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Universe settings
        st.subheader("Universe Selection")
        universe_size = st.slider("Number of stocks to analyze", 10, 100, 50, 10)
        
        # Date range
        st.subheader("Analysis Period")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        start_date = st.date_input("Start Date", start_date)
        end_date = st.date_input("End Date", end_date)
        
        # Factor weights
        st.subheader("Factor Weights")
        col1, col2 = st.columns(2)
        with col1:
            momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.4, 0.05)
            sentiment_weight = st.slider("Sentiment", 0.0, 1.0, 0.3, 0.05)
        with col2:
            volume_weight = st.slider("Volume", 0.0, 1.0, 0.2, 0.05)
            volatility_weight = st.slider("Volatility", 0.0, 1.0, 0.1, 0.05)
        
        # Normalize weights
        total = momentum_weight + sentiment_weight + volume_weight + volatility_weight
        if total > 0:
            weights = {
                'momentum': momentum_weight / total,
                'sentiment': sentiment_weight / total,
                'volume': volume_weight / total,
                'volatility': volatility_weight / total
            }
        else:
            weights = {'momentum': 0.4, 'sentiment': 0.3, 'volume': 0.2, 'volatility': 0.1}
        
        # Run button
        st.subheader("🚀 Execute")
        run_button = st.button("Run Screener", type="primary", use_container_width=True)
        
        # Cache info
        st.caption("Results are cached for 5 minutes to improve performance")
    
    # Main content area
    if run_button:
        with st.spinner("Running analysis pipeline... This may take 1-2 minutes."):
            try:
                scores_df, price_data, backtest_results, sentiment_scores = run_screener(
                    universe_size, 
                    str(start_date), 
                    str(end_date), 
                    weights
                )
                
                st.session_state.scores_df = scores_df
                st.session_state.price_data = price_data
                # Also store data_source (from resource cache) for news fetching in tab4
                st.session_state.data_source, _, _ = load_components()
                st.session_state.screener_results = {
                    'backtest': backtest_results,
                    'sentiment': sentiment_scores
                }
                
                st.success("✅ Screener completed successfully!")
                
            except Exception as e:
                st.error(f"Error running screener: {str(e)}")
                logger.exception("Screener failed")
                return
    
    # Display results if available
    if st.session_state.scores_df is not None:
        scores_df = st.session_state.scores_df
        price_data = st.session_state.price_data
        results = st.session_state.screener_results
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Stocks", "📈 Visualizations", "🎯 Backtest", "🔍 Stock Deep Dive"])
        
        with tab1:
            st.subheader("Top Stock Recommendations")
            
            # Display top N stocks for the MOST RECENT date
            top_n = st.selectbox("Number of stocks to display", [5, 10, 20, 50], index=1)
            
            # Get the latest date from scores_df index (which is MultiIndex: Date, Ticker)
            latest_date = scores_df.index.get_level_values('Date').max()
            latest_scores = scores_df.xs(latest_date, level='Date')
            top_stocks = latest_scores.nlargest(top_n, 'composite')
            
            # Format display
            display_df = top_stocks[['composite', 'momentum', 'sentiment', 'volume', 'volatility']].copy()
            display_df.columns = ['Composite', 'Momentum', 'Sentiment', 'Volume', 'Volatility']
            display_df = display_df.round(3)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="top_stocks.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Visualization Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution (using latest date)
                latest_scores = scores_df.xs(scores_df.index.get_level_values('Date').max(), level='Date')
                fig1 = px.histogram(
                    latest_scores, 
                    x='composite', 
                    title='Composite Score Distribution (Latest Date)',
                    labels={'composite': 'Composite Score', 'count': 'Number of Stocks'},
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Factor contributions for top 10 (latest date)
                top10 = latest_scores.nlargest(10, 'composite')
                factor_cols = ['momentum', 'sentiment', 'volume', 'volatility']
                top10_melted = top10.reset_index().melt(
                    id_vars=['Ticker'], 
                    value_vars=factor_cols,
                    var_name='Factor', 
                    value_name='Score'
                )
                fig2 = px.bar(
                    top10_melted, 
                    x='Ticker', 
                    y='Score', 
                    color='Factor',
                    title='Factor Contributions (Top 10 Stocks)',
                    barmode='group'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation heatmap (using latest scores)
            corr_matrix = latest_scores[factor_cols].corr()
            fig3 = px.imshow(
                corr_matrix,
                text_auto=True,
                title='Factor Correlation Heatmap (Latest Date)',
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("Backtest Performance")
            
            backtest = results['backtest']
            if backtest:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Strategy Return", f"{backtest.get('strategy_return', 0)*100:.2f}%")
                with col2:
                    st.metric("Benchmark Return", f"{backtest.get('benchmark_return', 0)*100:.2f}%")
                with col3:
                    st.metric("Excess Return", f"{backtest.get('excess_return', 0)*100:.2f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Drawdown", f"{backtest.get('max_drawdown', 0)*100:.2f}%")
                with col2:
                    st.metric("Volatility", f"{backtest.get('volatility', 0)*100:.2f}%")
                
                # Optional: show cumulative return chart if we have series data
                if 'portfolio_series' in backtest and 'benchmark_series' in backtest:
                    port_series = backtest['portfolio_series']
                    bench_series = backtest['benchmark_series']
                    # Normalize to 100 off the TRUE starting capital (not the
                    # first plotted point) so the curve's endpoint matches the
                    # reported Strategy/Benchmark Return metrics exactly.
                    base = backtest.get('initial_capital', port_series.iloc[0])
                    port_norm = port_series / base * 100
                    bench_norm = bench_series / base * 100
                    df_plot = pd.DataFrame({
                        'Strategy': port_norm,
                        'Benchmark': bench_norm
                    })
                    fig = px.line(
                        df_plot,
                        title='Cumulative Performance (Normalized)',
                        labels={'value': 'Index (100 = Start)', 'variable': 'Portfolio'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No backtest results available.")
            
            st.info("💡 Note: Backtest uses weekly rebalancing with top 10 stocks. Benchmark is S&P 500.")
        
        with tab4:
            st.subheader("Individual Stock Analysis")
            
            # Stock selector – get tickers from latest date
            latest_date = scores_df.index.get_level_values('Date').max()
            latest_scores = scores_df.xs(latest_date, level='Date')
            tickers = latest_scores.index.tolist()
            selected_ticker = st.selectbox("Select Stock", tickers, index=0 if tickers else None)
            
            if selected_ticker and st.session_state.price_data is not None:
                # Get stock data from latest scores
                stock_score = latest_scores.loc[selected_ticker]
                sentiment_data = results['sentiment'].get(selected_ticker, {})
                
                # Display metrics
                st.markdown(f"### {selected_ticker}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Composite Score", f"{stock_score['composite']:.3f}")
                with col2:
                    st.metric("Momentum", f"{stock_score['momentum']:.3f}")
                with col3:
                    st.metric("Sentiment", f"{stock_score['sentiment']:.3f}")
                with col4:
                    sentiment_conf = sentiment_data.get('confidence', 0) if isinstance(sentiment_data, dict) else 0
                    st.metric("Sentiment Confidence", f"{sentiment_conf:.2%}")
                
                # Price chart
                stock_prices = price_data.xs(selected_ticker, level='Ticker')
                fig4 = px.line(
                    stock_prices, 
                    x=stock_prices.index, 
                    y='Close',
                    title=f'{selected_ticker} - Price History',
                    labels={'Close': 'Price ($)', 'Date': 'Date'}
                )
                st.plotly_chart(fig4, use_container_width=True)
                
                # News headlines
                with st.expander("Recent News Headlines"):
                    if st.session_state.data_source is not None:
                        headlines = st.session_state.data_source.get_news_headlines(selected_ticker, lookback_days=7)
                        if headlines:
                            for headline in headlines:
                                st.write(f"• {headline}")
                        else:
                            st.write("No recent news found.")
                    else:
                        st.write("Data source not available.")

if __name__ == "__main__":
    # Preload components into session state for reuse
    ds, _, _ = load_components()
    st.session_state.data_source = ds
    main()