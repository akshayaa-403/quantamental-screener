from dotenv import load_dotenv
load_dotenv()
import logging
import time
import traceback

import requests
import streamlit as st

st.set_page_config(
    page_title="Quantamental Screener",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on phones, where it also holds the nav
)

from ui import pages, stages, state  # noqa: E402 - after set_page_config
from ui.state import FACTOR_COLS, PERIOD_OPTIONS  # noqa: E402
from ui.theme import fmt_date, weights_bar_html  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The theme lives in .streamlit/config.toml; this only tightens the top padding,
# adds tabular figures, and shrinks the Backtest page's secondary stats.
CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 4.5rem; }
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] { font-variant-numeric: tabular-nums; }
.st-key-secondary_stats [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
"""

RATE_LIMITED = "Yahoo Finance is rate-limiting requests from this server. Wait a minute, then run again."
NETWORK = "Couldn't reach a data provider (network error). Check the connection, then run again."


def friendly_message(exc: Exception) -> str:
    last = getattr(exc, "last_attempt", None)  # tenacity.RetryError wraps the real error
    if last is not None and last.exception() is not None:
        exc = last.exception()
    if isinstance(exc, stages.ScreenerError):
        return str(exc)
    text = f"{type(exc).__name__} {exc}".lower()
    if "ratelimit" in text or "rate limit" in text or "too many requests" in text or "429" in text:
        return RATE_LIMITED
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
        return NETWORK
    return f"The screener stopped with an unexpected error ({type(exc).__name__}). Details below."


def run_pipeline(params: dict):
    """Run every stage under a live status panel; store results or the error in session state."""
    started = time.perf_counter()
    start, end, weights = params["start_date"], params["end_date"], params["weights"]
    with st.status("Running the screener…", expanded=True) as status:
        try:
            status.update(label="Selecting universe…")
            st.write(":material/public: Selecting universe")
            tickers = stages.select_universe(params["universe_size"])

            status.update(label=f"Downloading prices ({len(tickers)} tickers)…")
            st.write(f":material/download: Downloading prices ({len(tickers)} tickers)")
            prices = stages.download_prices(tuple(tickers), start, end)

            if params["use_mock"]:
                st.write(":material/newspaper: Fetching news: skipped (mock sentiment)")
                news_tickers = ()
                in_data = tuple(prices.index.get_level_values("Ticker").unique())
                sentiment = stages.mock_sentiment(in_data)
            else:
                news_tickers = tuple(tickers[: stages.LIVE_SENTIMENT_LIMIT])
                status.update(label=f"Fetching news ({len(news_tickers)} tickers)…")
                st.write(f":material/newspaper: Fetching news ({len(news_tickers)} tickers)")
                try:
                    sentiment = stages.live_sentiment(news_tickers)
                except stages.NoHeadlines:
                    sentiment = {}  # every name is scored neutral; a notice says so
                    st.write(":material/warning: No headlines came back")

            status.update(label="Scoring…")
            st.write(":material/functions: Scoring")
            scores = stages.score(prices, sentiment, weights)

            status.update(label="Backtesting…")
            st.write(":material/monitoring: Backtesting")
            backtest = stages.backtest(prices, weights, start)
        except Exception as exc:  # noqa: BLE001 - surface any pipeline error to the user
            logger.exception("Screener failed")
            status.update(label="Screener failed", state="error", expanded=False)
            st.session_state.last_error = (friendly_message(exc), traceback.format_exc())
            return
        status.update(label=f"Screener finished in {time.perf_counter() - started:.0f}s",
                      state="complete", expanded=False)

    st.session_state.results = {
        "scores": scores,
        "prices": prices,
        "backtest": backtest,
        "sentiment": sentiment,
        "news_tickers": news_tickers,
        "universe": tuple(tickers),
        "as_of": prices.index.get_level_values("Date").max(),
        "params": params,
    }
    st.session_state.last_run_params = params
    st.session_state.last_error = None
    st.session_state.run_id += 1
    n = prices.index.get_level_values("Ticker").nunique()
    st.toast(f"Screener finished: prices for {n} stocks", icon=":material/check_circle:")


def render_sidebar():
    """Settings. Widgets only change session state; the pipeline runs on Run screener."""
    ss = st.session_state
    with st.sidebar:
        st.markdown("### :material/tune: Settings")
        st.slider("Universe size", 10, 100, step=10, key="w_universe",
                  help="The N largest S&P 500 names by market cap that pass the market-cap and volume floors.")

        st.segmented_control("Backtest period", PERIOD_OPTIONS, key="w_period", on_change=state.keep_period,
                             help="Trading window for the backtest. About 400 extra days are downloaded "
                                  "before it so the factors can warm up.")
        valid = True
        if ss.w_period == "Custom":
            today = state._today()
            c1, c2 = st.columns(2)
            c1.date_input("Start", key="w_start", max_value=today, format="YYYY-MM-DD")
            c2.date_input("End", key="w_end", max_value=today, format="YYYY-MM-DD")
            if ss.w_end <= ss.w_start:
                st.error("End date must be after the start date.", icon=":material/event_busy:")
                valid = False
        start, end = state.period_dates()
        st.caption(f"{start:%Y-%m-%d} → {end:%Y-%m-%d} (end date excluded)")

        st.markdown("**Factor weights**")
        c1, c2 = st.columns(2)
        for col, factor in zip((c1, c1, c2, c2), FACTOR_COLS):  # stacks in factor order on phones
            col.slider(factor.capitalize(), 0.0, 1.0, step=0.05, key=f"w_{factor}")
        weights, fallback = state.normalized_weights()
        st.markdown(weights_bar_html(weights), unsafe_allow_html=True)
        if fallback:
            st.warning("All weights are 0, so the default weights are used.", icon=":material/warning:")
        st.button("Reset to defaults", on_click=state.reset_weights, type="tertiary", icon=":material/restart_alt:")

        st.toggle("Mock sentiment", key="w_mock",
                  help="On: seeded random sentiment for every name. Fast, offline and reproducible, but not "
                       f"real. Off: live news sentiment for the {stages.LIVE_SENTIMENT_LIMIT} largest names, "
                       "which adds a few seconds per name on a cold cache. Prices are always real.")

        st.divider()
        st.button("Run screener", type="primary", icon=":material/play_arrow:", width="stretch",
                  on_click=state.request_run, disabled=not valid)
        st.caption("Results are cached for 5 minutes.")
    return state.current_params(), valid


def render_header():
    results = st.session_state.results
    st.markdown("### :material/query_stats: Quantamental Screener")
    if results is None:
        chips = ":gray-badge[Not run yet]"
    else:
        sentiment = (":orange-badge[:material/science: Mock sentiment]" if results["params"]["use_mock"]
                     else ":blue-badge[:material/newspaper: Live sentiment]")
        chips = (f"{sentiment} :gray-badge[:material/calendar_today: Prices as of {fmt_date(results['as_of'])}] "
                 f":gray-badge[:material/stacks: Universe: {len(results['universe'])}]")
    st.markdown(
        "Ranks large-cap US stocks on momentum, volume, volatility and news sentiment, then backtests the "
        f"price factors against SPY. &nbsp; {chips}"
    )


def render_notices(params: dict, valid: bool):
    ss = st.session_state
    if ss.last_error:
        message, details = ss.last_error
        st.error(message, icon=":material/error:")
        with st.expander("Error details"):
            st.code(details, language="text")
    if ss.results is not None and not ss.results["params"]["use_mock"] and not ss.results["sentiment"]:
        st.warning("No live headlines could be fetched, so every stock's sentiment is blank (a z-score of 0 in "
                   "the composite). The news source may be down; try again later or use mock sentiment.",
                   icon=":material/newspaper:")
    if ss.results is not None and not state.same_settings(params, ss.last_run_params):
        msg, action = st.columns([4, 1], vertical_alignment="center")
        msg.warning("Settings changed — results are from the previous run.", icon=":material/history:")
        action.button("Rerun", type="primary", icon=":material/refresh:", width="stretch",
                      on_click=state.request_run, disabled=not valid, key="stale_rerun")


def main():
    state.init_state()
    st.html(CSS)
    params, valid = render_sidebar()
    header = st.container()
    if st.session_state.run_requested:
        st.session_state.run_requested = False
        if valid:
            run_pipeline(params)
    with header:
        render_header()
    render_notices(params, valid)
    st.navigation(pages.PAGES, position="top").run()


if __name__ == "__main__":
    main()
