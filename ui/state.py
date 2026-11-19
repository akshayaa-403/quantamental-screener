"""Session state: every key the dashboard uses, initialized in one place."""
from datetime import datetime, timedelta

import streamlit as st

from config.settings import get_settings

FACTOR_COLS = ("momentum", "sentiment", "volume", "volatility")
FACTOR_MEANING = {
    "momentum": "How much the stock rose over the past year, skipping the latest month.",
    "sentiment": "How positive or negative the last week's news headlines about the company are.",
    "volume": "Whether heavy trading days lately came with rising or falling prices (buying vs selling pressure).",
    "volatility": "How calm the stock's daily price swings were over the last 3 months; calmer scores higher.",
}
PERIOD_DAYS = {"6M": 182, "1Y": 365, "2Y": 730, "3Y": 1095}
PERIOD_OPTIONS = [*PERIOD_DAYS, "Custom"]
DEFAULT_PERIOD = "1Y"

# App state that is not tied to a widget.
STATE_DEFAULTS = {
    "results": None,          # outputs of the last successful run (see streamlit_app.run_pipeline)
    "last_run_params": None,  # the params that produced `results`
    "selected_ticker": None,  # ticker shared by Overview, Rankings and Stock detail
    "run_requested": False,   # set by any Run button, consumed by the next script run
    "last_error": None,       # (friendly message, traceback) of the last failed run
    "run_id": 0,              # bumped on every successful run; resets table selections
}
# Widgets on the Rankings page.
PAGE_WIDGET_DEFAULTS = {"rank_search": "", "rank_top_n": "10"}
# Stock detail's ticker selectbox uses the key "detail_ticker", synced from selected_ticker.


def _today():
    return datetime.now().date()


def widget_defaults() -> dict:
    """Default value of every sidebar widget, keyed by widget key."""
    s = get_settings()
    today = _today()
    return {
        "w_universe": s.universe.size,
        "w_period": DEFAULT_PERIOD,
        "w_start": today - timedelta(days=PERIOD_DAYS[DEFAULT_PERIOD]),
        "w_end": today,
        "w_momentum": s.weights.momentum,
        "w_sentiment": s.weights.sentiment,
        "w_volume": s.weights.volume,
        "w_volatility": s.weights.volatility,
        "w_mock": False,
    }


def init_state():
    """Run at the top of every script run: fill in missing keys and keep widget values.

    Streamlit drops a widget's value when the widget isn't rendered (another page,
    a hidden date picker) or when st.switch_page changes page. Each widget value is
    mirrored into a plain "_keep_" key and restored from it, so settings survive.
    """
    ss = st.session_state
    for key, value in STATE_DEFAULTS.items():
        ss.setdefault(key, value)
    for key, value in {**widget_defaults(), **PAGE_WIDGET_DEFAULTS}.items():
        if key in ss:
            ss[f"_keep_{key}"] = ss[key]
        else:
            ss[key] = ss.get(f"_keep_{key}", value)


def default_weights() -> dict:
    w = get_settings().weights
    return {c: getattr(w, c) for c in FACTOR_COLS}


def normalized_weights():
    """(weights, used_fallback): slider weights rescaled to sum to 1, or the defaults if all are 0."""
    ss = st.session_state
    m, s, v, vol = ss.w_momentum, ss.w_sentiment, ss.w_volume, ss.w_volatility
    total = m + s + v + vol  # same summation order as before, so the floats are bit-identical
    if total > 0:
        return {"momentum": m / total, "sentiment": s / total, "volume": v / total, "volatility": vol / total}, False
    return default_weights(), True


def period_dates():
    """(start, end) dates for the chosen period preset or the custom range."""
    ss = st.session_state
    if ss.w_period == "Custom":
        return ss.w_start, ss.w_end
    today = _today()
    return today - timedelta(days=PERIOD_DAYS[ss.w_period]), today


def current_params() -> dict:
    """The run parameters implied by the sidebar right now (hashable values only)."""
    start, end = period_dates()
    weights, _ = normalized_weights()
    return {
        "universe_size": int(st.session_state.w_universe),
        "start_date": str(start),
        "end_date": str(end),
        "weights": tuple(weights.items()),
        "use_mock": bool(st.session_state.w_mock),
        "period": st.session_state.w_period,
    }


def same_settings(a, b) -> bool:
    """Whether two param sets come from the same sidebar choices.

    Presets are compared by label, so a session left open past midnight (when
    "1Y" moves forward a day) is not flagged as having changed settings.
    """
    if a is None or b is None:
        return a is b
    if a["period"] != "Custom" and a["period"] == b["period"]:
        a, b = {**a, "start_date": None, "end_date": None}, {**b, "start_date": None, "end_date": None}
    return a == b


# --- widget callbacks -------------------------------------------------------

def request_run():
    st.session_state.run_requested = True


def reset_weights():
    for c, value in default_weights().items():
        st.session_state[f"w_{c}"] = value


def request_demo_run():
    """Default settings with mock sentiment, then run."""
    for key, value in widget_defaults().items():
        st.session_state[key] = value
    st.session_state.w_mock = True
    request_run()


def keep_period():
    """A segmented control can be deselected; fall back to the default preset."""
    if st.session_state.w_period is None:
        st.session_state.w_period = DEFAULT_PERIOD
