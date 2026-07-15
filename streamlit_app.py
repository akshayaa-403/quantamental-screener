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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACTOR_COLS = ["momentum", "sentiment", "volume", "volatility"]
# Palette that reads well in both light and dark themes.
STRATEGY_COLOR = "#22c55e"
BENCHMARK_COLOR = "#94a3b8"
ACCENT = "#6366f1"

# Page configuration
st.set_page_config(
    page_title="Quantamental Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware styling: a gradient hero and softened containers that work in
# both light and dark mode (no hard-coded light-only backgrounds).
st.markdown(
    """
<style>
    .hero {
        background: linear-gradient(120deg, #6366f1 0%, #22c55e 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 0.9rem;
        margin-bottom: 1.2rem;
        color: #ffffff;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #ffffff; }
    .hero p  { margin: 0.35rem 0 0 0; font-size: 1.02rem; opacity: 0.92; }
    /* Metric cards adapt to theme via Streamlit's own variables. */
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.18);
        padding: 0.75rem 1rem;
        border-radius: 0.7rem;
    }
    div[data-testid="stMetricLabel"] { opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
for _key in ("screener_results", "price_data", "scores_df", "data_source"):
    if _key not in st.session_state:
        st.session_state[_key] = None


@st.cache_resource(ttl=3600)
def load_components():
    """Load heavy components once and cache them."""
    data_source = YahooFinanceSource()
    sentiment_model = EnsembleSentiment()
    factors = [MomentumFactor(), SentimentFactor(), VolumeFactor(), VolatilityFactor()]
    return data_source, sentiment_model, factors


@st.cache_data(ttl=300)  # 5 minute cache
def run_screener(universe_size, start_date, end_date, weights, use_mock_sentiment):
    """Run the complete screener pipeline with caching."""
    data_source, sentiment_model, factors = load_components()

    # Update factor weights on the (cached) factor instances.
    for factor in factors:
        if factor.name in weights:
            factor.weight = weights[factor.name]

    # Universe selection
    tickers = UniverseSelector({"size": universe_size}).select()

    # Data collection
    price_data = DataCollector(data_source).collect(tickers, start_date, end_date)

    if use_mock_sentiment:
        # Seeded mock sentiment: fast, offline, reproducible.
        rng = np.random.default_rng(42)
        tickers_in_data = price_data.index.get_level_values("Ticker").unique()
        sentiment_scores = {t: {"score": float(rng.uniform(-0.8, 0.8))} for t in tickers_in_data}
    else:
        # Real sentiment for the first 20 tickers; missing names stay neutral (0.0).
        limited_tickers = tickers[: min(20, len(tickers))]
        news_by_ticker = {}
        for ticker in limited_tickers:
            headlines = data_source.get_news_headlines(ticker, lookback_days=7)
            if headlines:
                news_by_ticker[ticker] = headlines
        sentiment_scores = sentiment_model.batch_analyze(news_by_ticker) if news_by_ticker else {}

    # Scoring
    scorer = ScoringEngine(factors)
    scores_df = scorer.compute(price_data, sentiment_scores, cross_sectional_normalize=True)

    # Backtest
    backtest_results = BacktestEngine().run(scorer, price_data, sentiment_scores)

    return scores_df, price_data, backtest_results, sentiment_scores


def render_sidebar():
    """Render the configuration sidebar and return the chosen settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Universe")
        universe_size = st.slider("Number of stocks to analyze", 10, 100, 50, 10)

        st.subheader("Analysis period")
        today = datetime.now().date()
        start_date = st.date_input("Start date", today - timedelta(days=365))
        end_date = st.date_input("End date", today)

        st.subheader("Factor weights")
        c1, c2 = st.columns(2)
        with c1:
            momentum_w = st.slider("Momentum", 0.0, 1.0, 0.4, 0.05)
            sentiment_w = st.slider("Sentiment", 0.0, 1.0, 0.3, 0.05)
        with c2:
            volume_w = st.slider("Volume", 0.0, 1.0, 0.2, 0.05)
            volatility_w = st.slider("Volatility", 0.0, 1.0, 0.1, 0.05)

        total = momentum_w + sentiment_w + volume_w + volatility_w
        if total > 0:
            weights = {
                "momentum": momentum_w / total,
                "sentiment": sentiment_w / total,
                "volume": volume_w / total,
                "volatility": volatility_w / total,
            }
        else:
            weights = {"momentum": 0.4, "sentiment": 0.3, "volume": 0.2, "volatility": 0.1}

        # Live preview of the normalized weights the screen will actually use.
        st.caption("Normalized weights (sum = 100%)")
        weight_df = pd.DataFrame(
            {"Weight": [weights[c] for c in FACTOR_COLS]},
            index=[c.capitalize() for c in FACTOR_COLS],
        )
        st.dataframe(
            weight_df,
            width="stretch",
            column_config={
                "Weight": st.column_config.ProgressColumn(
                    "Weight", format="percent", min_value=0.0, max_value=1.0
                )
            },
        )

        st.subheader("Sentiment source")
        use_mock_sentiment = st.toggle(
            "Use mock sentiment (fast offline demo)",
            value=False,
            help="On: seeded random scores, no news calls. Off: real news-based sentiment.",
        )

        st.divider()
        run_button = st.button("🚀 Run Screener", type="primary", width="stretch")
        st.caption("Results are cached for 5 minutes.")

    return universe_size, start_date, end_date, weights, use_mock_sentiment, run_button


def render_top_stocks(scores_df):
    st.subheader("Top stock recommendations")
    top_n = st.selectbox("Number of stocks to display", [5, 10, 20, 50], index=1)

    latest_date = scores_df.index.get_level_values("Date").max()
    latest_scores = scores_df.xs(latest_date, level="Date")
    top_stocks = latest_scores.nlargest(top_n, "composite")

    # KPI strip
    k1, k2, k3 = st.columns(3)
    k1.metric("Stocks analyzed", f"{len(latest_scores):,}")
    k2.metric("Avg composite (top)", f"{top_stocks['composite'].mean():.3f}")
    k3.metric("Top pick", str(top_stocks.index[0]) if len(top_stocks) else "—")

    disp = top_stocks[["rank"] + FACTOR_COLS].copy()
    disp.insert(1, "composite", top_stocks["composite"])
    disp.columns = ["Rank", "Composite", "Momentum", "Sentiment", "Volume", "Volatility"]
    disp["Rank"] = disp["Rank"].astype(int)

    cmin, cmax = float(disp["Composite"].min()), float(disp["Composite"].max())
    if cmin == cmax:
        cmax = cmin + 1e-9

    st.dataframe(
        disp,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Composite": st.column_config.ProgressColumn(
                "Composite", format="%.3f", min_value=cmin, max_value=cmax
            ),
            "Momentum": st.column_config.NumberColumn("Momentum", format="%.3f"),
            "Sentiment": st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "Volume": st.column_config.NumberColumn("Volume", format="%.3f"),
            "Volatility": st.column_config.NumberColumn("Volatility", format="%.3f"),
        },
    )

    st.download_button(
        label="📥 Download as CSV",
        data=disp.to_csv().encode("utf-8"),
        file_name="top_stocks.csv",
        mime="text/csv",
    )


def render_visualizations(scores_df):
    st.subheader("Visualizations")
    latest_scores = scores_df.xs(scores_df.index.get_level_values("Date").max(), level="Date")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(
            latest_scores,
            x="composite",
            nbins=20,
            title="Composite score distribution",
            labels={"composite": "Composite score"},
            color_discrete_sequence=[ACCENT],
        )
        fig1.update_layout(margin=dict(t=50, b=10), bargap=0.05)
        st.plotly_chart(fig1, width="stretch")

    with c2:
        top10 = latest_scores.nlargest(10, "composite")
        melted = top10.reset_index().melt(
            id_vars=["Ticker"], value_vars=FACTOR_COLS, var_name="Factor", value_name="Score"
        )
        fig2 = px.bar(
            melted,
            x="Ticker",
            y="Score",
            color="Factor",
            barmode="group",
            title="Factor contributions (top 10)",
        )
        fig2.update_layout(margin=dict(t=50, b=10), legend_title_text="")
        st.plotly_chart(fig2, width="stretch")

    corr = latest_scores[FACTOR_COLS].corr()
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        title="Factor correlation",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig3.update_layout(margin=dict(t=50, b=10))
    st.plotly_chart(fig3, width="stretch")


def render_backtest(backtest):
    st.subheader("Backtest performance")
    if not backtest:
        st.warning("No backtest results available.")
        return

    sr = backtest.get("strategy_return", 0) * 100
    br = backtest.get("benchmark_return", 0) * 100
    er = backtest.get("excess_return", 0) * 100
    sharpe = backtest.get("sharpe_ratio", 0)
    mdd = backtest.get("max_drawdown", 0) * 100
    vol = backtest.get("volatility", 0) * 100
    hit = backtest.get("hit_rate", 0) * 100
    final = backtest.get("final_value", 0)
    init = backtest.get("initial_capital", 0)

    with st.container(border=True):
        r1 = st.columns(4)
        r1[0].metric("Strategy return", f"{sr:.2f}%", delta=f"{er:+.2f}% vs benchmark")
        r1[1].metric("Benchmark return", f"{br:.2f}%")
        r1[2].metric("Sharpe ratio", f"{sharpe:.2f}", help=">1 good, >2 very good, >3 excellent")
        r1[3].metric("Hit rate", f"{hit:.1f}%", help="Share of days the strategy beat the benchmark")

        r2 = st.columns(4)
        r2[0].metric("Max drawdown", f"{mdd:.2f}%", help="Worst peak-to-trough loss")
        r2[1].metric("Volatility (ann.)", f"{vol:.2f}%")
        r2[2].metric("Final value", f"${final:,.0f}", delta=f"${final - init:,.0f}")
        r2[3].metric("Excess return", f"{er:+.2f}%")

    if "portfolio_series" in backtest and "benchmark_series" in backtest:
        port_series = backtest["portfolio_series"]
        bench_series = backtest["benchmark_series"]
        # Normalize off the TRUE starting capital so the curve's endpoint matches
        # the reported Strategy/Benchmark Return metrics exactly.
        base = backtest.get("initial_capital", port_series.iloc[0])
        df_plot = pd.DataFrame(
            {"Strategy": port_series / base * 100, "Benchmark": bench_series / base * 100}
        )
        fig = px.line(
            df_plot,
            title="Cumulative performance (start = 100)",
            labels={"value": "Index", "variable": "", "index": "Date"},
            color_discrete_map={"Strategy": STRATEGY_COLOR, "Benchmark": BENCHMARK_COLOR},
        )
        fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            margin=dict(t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, width="stretch")

    st.caption("💡 Weekly rebalancing into the top-ranked names. Benchmark: S&P 500 (SPY).")


def render_deep_dive(scores_df, price_data, results):
    st.subheader("Individual stock analysis")
    latest_date = scores_df.index.get_level_values("Date").max()
    latest_scores = scores_df.xs(latest_date, level="Date")
    tickers = latest_scores.index.tolist()
    if not tickers:
        st.info("No stocks available.")
        return

    selected_ticker = st.selectbox("Select stock", tickers, index=0)
    stock_score = latest_scores.loc[selected_ticker]
    sentiment_data = results["sentiment"].get(selected_ticker, {}) if results else {}

    st.markdown(f"### {selected_ticker}")
    cols = st.columns(4)
    # Deltas are relative to the universe average for context.
    cols[0].metric(
        "Composite",
        f"{stock_score['composite']:.3f}",
        delta=f"{stock_score['composite'] - latest_scores['composite'].mean():+.3f} vs avg",
    )
    cols[1].metric("Momentum", f"{stock_score['momentum']:.3f}")
    cols[2].metric("Sentiment", f"{stock_score['sentiment']:.3f}")
    conf = sentiment_data.get("confidence", 0) if isinstance(sentiment_data, dict) else 0
    cols[3].metric("Sentiment confidence", f"{conf:.0%}")

    if price_data is not None:
        stock_prices = price_data.xs(selected_ticker, level="Ticker")
        fig = px.line(
            stock_prices,
            x=stock_prices.index,
            y="Close",
            title=f"{selected_ticker} — price history",
            labels={"Close": "Price ($)", "x": "Date"},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Recent news headlines"):
        source = st.session_state.get("data_source")
        if source is not None:
            headlines = source.get_news_headlines(selected_ticker, lookback_days=7)
            if headlines:
                for headline in headlines:
                    st.write(f"• {headline}")
            else:
                st.write("No recent news found.")
        else:
            st.write("Data source not available.")


def render_empty_state():
    st.info("👈 Configure your screen in the sidebar, then click **Run Screener** to begin.")
    with st.container(border=True):
        st.markdown("#### How it works")
        st.markdown(
            "- **Momentum, Sentiment, Volume and (low) Volatility** factors are "
            "z-score standardized across stocks each day and blended with your "
            "weights into a **composite** score.\n"
            "- Stocks are ranked daily; the backtest rebalances weekly into the "
            "top names and compares against the S&P 500."
        )


def main():
    st.markdown(
        """
<div class="hero">
    <h1>📈 Quantamental Equity Screener</h1>
    <p>Multi-factor analysis combining technical indicators, sentiment, and risk.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    universe_size, start_date, end_date, weights, use_mock_sentiment, run_button = render_sidebar()

    if run_button:
        with st.spinner("Running analysis pipeline… this may take 1–2 minutes."):
            try:
                scores_df, price_data, backtest_results, sentiment_scores = run_screener(
                    universe_size, str(start_date), str(end_date), weights, use_mock_sentiment
                )
                st.session_state.scores_df = scores_df
                st.session_state.price_data = price_data
                st.session_state.data_source, _, _ = load_components()
                st.session_state.screener_results = {
                    "backtest": backtest_results,
                    "sentiment": sentiment_scores,
                }
                st.success("✅ Screener completed successfully!")
            except Exception as e:  # noqa: BLE001 - surface any pipeline error to the user
                st.error(f"Error running screener: {e}")
                logger.exception("Screener failed")
                return

    if st.session_state.scores_df is None:
        render_empty_state()
        return

    scores_df = st.session_state.scores_df
    price_data = st.session_state.price_data
    results = st.session_state.screener_results

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Top Stocks", "📈 Visualizations", "🎯 Backtest", "🔍 Stock Deep Dive"]
    )
    with tab1:
        render_top_stocks(scores_df)
    with tab2:
        render_visualizations(scores_df)
    with tab3:
        render_backtest(results["backtest"] if results else None)
    with tab4:
        render_deep_dive(scores_df, price_data, results)


if __name__ == "__main__":
    ds, _, _ = load_components()
    st.session_state.data_source = ds
    main()
