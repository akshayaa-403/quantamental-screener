#!/usr/bin/env python3
"""
Streamlit Web Interface for Quantamental Equity Screener
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# Custom modules
from config.settings import Config, config
from quantamental.universe import UniverseSelector
from quantamental.data_collector import DataCollector
from quantamental.sentiment import SentimentAnalyzer
from quantamental.scoring import ScoringEngine
from quantamental.visualizer import Visualizer
from quantamental.backtest import BacktestEngine
from quantamental.utils import generate_report, analyze_individual_stock
from utils.streamlit_cache import load_finbert, download_price_data, get_sp500_tickers_cached

# Page config
st.set_page_config(page_title="Quantamental Screener", layout="wide", page_icon="📈")
st.title("📊 Quantamental Equity Screener")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    universe_size = st.number_input("Universe Size (stocks)", min_value=10, max_value=100, value=50, step=10)
    top_n = st.number_input("Top N Stocks to Display", min_value=5, max_value=30, value=10, step=5)
    data_period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"], index=1)
    backtest_months = st.slider("Backtest Period (months)", 3, 24, 6)
    
    st.subheader("Scoring Weights")
    w_momentum = st.slider("Momentum Weight", 0.0, 1.0, config.MOMENTUM_WEIGHT, 0.05)
    w_sentiment = st.slider("Sentiment Weight", 0.0, 1.0, config.SENTIMENT_WEIGHT, 0.05)
    w_volume = st.slider("Volume Weight", 0.0, 1.0, config.VOLUME_WEIGHT, 0.05)
    w_volatility = st.slider("Volatility Weight", 0.0, 1.0, config.VOLATILITY_WEIGHT, 0.05)
    
    # Normalize weights to sum to 1
    total = w_momentum + w_sentiment + w_volume + w_volatility
    if total > 0:
        config.MOMENTUM_WEIGHT = w_momentum / total
        config.SENTIMENT_WEIGHT = w_sentiment / total
        config.VOLUME_WEIGHT = w_volume / total
        config.VOLATILITY_WEIGHT = w_volatility / total
    
    st.caption(f"Weights normalized: M={config.MOMENTUM_WEIGHT:.2f}, S={config.SENTIMENT_WEIGHT:.2f}, V={config.VOLUME_WEIGHT:.2f}, Vol={config.VOLATILITY_WEIGHT:.2f}")
    
    run_btn = st.button("🚀 Run Screener", type="primary", use_container_width=True)

# Main logic
if run_btn:
    # Update config with user choices
    config.UNIVERSE_SIZE = universe_size
    config.TOP_N_STOCKS = top_n
    config.DATA_PERIOD = data_period
    
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    try:
        # 1. Universe selection
        status_text.text("1/8: Selecting stock universe...")
        progress_bar.progress(10)
        universe_selector = UniverseSelector(config)
        # Use cached tickers for speed
        tickers = get_sp500_tickers_cached()
        universe = universe_selector.filter_universe(tickers)
        st.session_state['universe'] = universe
        st.session_state['sector_info'] = universe_selector.sector_info
        status_text.text(f"Universe: {len(universe)} stocks")
        
        # 2. Data download (cached per ticker)
        status_text.text("2/8: Downloading price data...")
        progress_bar.progress(20)
        price_data = {}
        for i, ticker in enumerate(universe[:config.UNIVERSE_SIZE]):
            price_data[ticker] = download_price_data(ticker, period=config.DATA_PERIOD)
        data_collector = DataCollector(config)
        data_collector.price_data = price_data
        data_collector.process_all_technical_data()
        
        # 3. Sentiment analysis (limit to first 20 for speed)
        status_text.text("3/8: Analyzing news sentiment...")
        progress_bar.progress(40)
        sentiment_analyzer = SentimentAnalyzer(config)
        # Replace FinBERT with cached version
        sentiment_analyzer.finbert = load_finbert()
        sentiment_data = sentiment_analyzer.analyze_all_stocks(universe[:20])
        
        # 4. Scoring
        status_text.text("4/8: Scoring stocks...")
        progress_bar.progress(60)
        scoring_engine = ScoringEngine(config, data_collector, sentiment_data, universe_selector)
        available = list(set(price_data.keys()) & set(sentiment_data.keys()))
        scores_df = scoring_engine.score_all_stocks(available)
        top_stocks = scoring_engine.get_top_stocks(scores_df, n=config.TOP_N_STOCKS)
        
        # 5. Visualizations (store in session)
        status_text.text("5/8: Generating charts...")
        progress_bar.progress(70)
        visualizer = Visualizer(config, price_data)
        
        # Create figures
        fig_dist = visualizer.create_score_distribution(scores_df)
        fig_sector = visualizer.create_sector_breakdown(scores_df)
        fig_corr = visualizer.create_factor_correlation(scores_df)
        fig_top = visualizer.create_top_stocks_chart(top_stocks)
        fig_factors = visualizer.create_factor_breakdown(top_stocks)
        
        # 6. Backtest
        status_text.text("6/8: Running backtest...")
        progress_bar.progress(80)
        backtest_engine = BacktestEngine(config)
        backtest_universe = list(price_data.keys())[:10]
        hist_data = backtest_engine.get_historical_universe(backtest_months, backtest_universe)
        backtest_results = backtest_engine.run_backtest(hist_data)
        backtest_fig = backtest_engine.create_backtest_visualization(backtest_results)
        
        # 7. Individual analysis for top 3
        status_text.text("7/8: Preparing individual analysis...")
        progress_bar.progress(90)
        ind_analysis = {}
        for ticker in top_stocks['ticker'].head(3):
            analysis = analyze_individual_stock(ticker, scores_df, data_collector, sentiment_data, universe_selector)
            # capture the print output? We'll just store the row for now.
            ind_analysis[ticker] = scores_df[scores_df['ticker'] == ticker].iloc[0].to_dict()
        
        # 8. Report
        status_text.text("8/8: Finalizing report...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Save to session state
        st.session_state['scores_df'] = scores_df
        st.session_state['top_stocks'] = top_stocks
        st.session_state['backtest_results'] = backtest_results
        st.session_state['ind_analysis'] = ind_analysis
        st.session_state['figures'] = {
            'score_dist': fig_dist,
            'sector': fig_sector,
            'corr': fig_corr,
            'top': fig_top,
            'factors': fig_factors,
            'backtest': backtest_fig
        }
        st.session_state['run_complete'] = True
        st.success("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"Error during screening: {str(e)}")
        st.stop()

# Display results if available
if st.session_state.get('run_complete', False):
    st.markdown("## 📈 Results Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Overview", "📉 Factor Analysis", "🔍 Top Stocks", "⏱️ Backtest"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state['figures']['score_dist'], use_container_width=True)
        with col2:
            st.plotly_chart(st.session_state['figures']['sector'], use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state['figures']['corr'], use_container_width=True)
        with col2:
            st.plotly_chart(st.session_state['figures']['factors'], use_container_width=True)
            
        # Factor correlation table
        with st.expander("View Factor Correlation Matrix (numeric)"):
            factor_cols = ['momentum_score', 'sentiment_score', 'volume_score', 'volatility_score']
            corr_matrix = st.session_state['scores_df'][factor_cols].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', axis=None))
    
    with tab3:
        st.subheader("🏆 Top Recommendations")
        st.dataframe(
            st.session_state['top_stocks'][['rank', 'ticker', 'composite_score', 'momentum_score', 'sentiment_score', 'volume_score', 'volatility_score', 'sector']],
            use_container_width=True,
            hide_index=True
        )
        st.plotly_chart(st.session_state['figures']['top'], use_container_width=True)
        
        # Individual stock dropdown
        st.subheader("🔎 Individual Stock Deep Dive")
        selected_ticker = st.selectbox("Select a ticker", st.session_state['top_stocks']['ticker'].tolist())
        if selected_ticker:
            stock_row = st.session_state['scores_df'][st.session_state['scores_df']['ticker'] == selected_ticker].iloc[0]
            st.json({
                "Ticker": selected_ticker,
                "Composite Score": round(stock_row['composite_score'], 3),
                "Momentum Score": round(stock_row['momentum_score'], 3),
                "Sentiment Score": round(stock_row['sentiment_score'], 3),
                "Volume Score": round(stock_row['volume_score'], 3),
                "Volatility Score": round(stock_row['volatility_score'], 3),
                "Sector": stock_row['sector']
            })
    
    with tab4:
        st.plotly_chart(st.session_state['figures']['backtest'], use_container_width=True)
        # Summary metrics
        res = st.session_state['backtest_results']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{res['total_return']:.2%}")
        col2.metric("Benchmark Return", f"{res['benchmark_return']:.2%}")
        col3.metric("Excess Return", f"{res['excess_return']:.2%}")
        col4.metric("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
        
    st.markdown("---")
    st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results.")

else:
    st.info("👈 Configure settings in the sidebar and click 'Run Screener' to begin.")