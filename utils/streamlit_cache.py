# utils/streamlit_cache.py
import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline

@st.cache_resource
def load_finbert():
    """Load FinBERT model once and cache it."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

@st.cache_data(ttl=3600)  # cache for 1 hour
def download_price_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Download price data with caching."""
    data = yf.download(ticker, period=period, progress=False)
    return data

@st.cache_data(ttl=86400)  # cache for 1 day
def get_sp500_tickers_cached():
    """Fetch S&P 500 tickers with caching (uses universe selector)."""
    from quantamental.universe import UniverseSelector
    from config.settings import config
    selector = UniverseSelector(config)
    return selector.get_sp500_tickers()