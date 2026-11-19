"""Page renderers: Overview, Rankings, Backtest and Stock detail."""
import re

import pandas as pd
import streamlit as st

from config.settings import get_settings
from ui import stages
from ui.state import FACTOR_COLS, request_demo_run
from ui.theme import (
    FACTOR_LABELS, composite_histogram, correlation_heatmap, equity_figure, factor_bar_figure,
    fmt_date, fmt_p, fmt_return, fmt_share, fmt_usd, fmt_z, md_usd, mini_factor_bars_html, naive_index,
    ordinal, p_value_reading, price_volume_figure, show_chart, z_cell_css,
)

README_BACKTEST_URL = (
    "https://github.com/akshayaa-403/quantamental-screener"
    "#how-the-backtest-works-and-what-it-cant-tell-you"
)
FACTOR_HELP = {
    "momentum": "12-minus-1 month return, z-scored across today's universe and clipped at ±3.",
    "volume": "10-day average of (volume ÷ 20-day average volume) × sign of the day's return: sustained "
              "buying vs selling pressure; z-scored.",
    "volatility": "Negative 60-day annualized volatility, z-scored: higher means a calmer stock.",
}
SENTIMENT_HELP_LIVE = (
    "News sentiment z-score from the last week's headlines naming the company, shrunk toward neutral when "
    "there are few headlines. Blank = no recent headlines; those names get a sentiment z-score of 0 in the "
    "composite, i.e. the average."
)
PIT_RANK_HELP = (
    "Rank on momentum, volume and volatility only, with your weights: the part of today's ranking the "
    "backtest actually tested. A big gap from Rank means sentiment, which the backtest can't test, is "
    "driving that name's position."
)
SENTIMENT_HELP_MOCK = "Seeded mock sentiment z-score (demo mode, not real news)."


# --- shared helpers ----------------------------------------------------------
def latest_scores(results) -> pd.DataFrame:
    scores = results["scores"]
    return scores.xs(scores.index.get_level_values("Date").max(), level="Date")


def ranked_scores(latest: pd.DataFrame) -> pd.DataFrame:
    """Names with a composite, best first (stable, so ties keep ticker order like nlargest)."""
    return latest.dropna(subset=["composite"]).sort_values("composite", ascending=False, kind="mergesort")


def tickers_with_sentiment(results) -> set:
    """Tickers that actually have a sentiment score (mock or fetched news)."""
    def numeric(v):
        v = v.get("score") if isinstance(v, dict) else v
        return isinstance(v, (int, float)) and not isinstance(v, bool) and not pd.isna(v)
    return {t for t, v in results["sentiment"].items() if numeric(v)}


def _md_escape(text: str) -> str:
    return re.sub(r"([\\$\[\]*_`])", r"\\\1", text)


def headline_markdown(item) -> str:
    """One bullet: the headline linked to the article when there is a URL, then an inline outlet chip."""
    if isinstance(item, str):
        item = {"title": item}
    title = _md_escape(item.get("title") or "")
    url = item.get("url")
    line = f"[{title}]({url.replace(')', '%29').replace(' ', '%20')})" if url else title
    tag = item.get("tag")
    chip = f" :gray-badge[{_md_escape(tag)}]" if tag else ""
    return f"- {line}{chip}"


def sentiment_source(results, ticker) -> str:
    if ticker in tickers_with_sentiment(results):
        return "mock" if results["params"]["use_mock"] else "live"
    if ticker in results["news_tickers"]:
        return "no headlines"
    return "not fetched"


def backtest_problem(results):
    """Why the backtest can't be read at face value, or None."""
    if not results["backtest"]:
        return "No backtest: not enough price history in this window for the factors to warm up."
    weights = dict(results["params"]["weights"])
    if not any(weights[c] > 0 for c in FACTOR_COLS if c != "sentiment"):
        return ("Backtest not meaningful: it excludes sentiment, and every other factor weight is 0, so all "
                "stocks score the same. Give momentum, volume or volatility some weight.")
    return None


def open_page(page, ticker=None):
    if ticker is not None:
        st.session_state.selected_ticker = ticker
    st.switch_page(page)


def kpi(col, label, value, note=None, delta=None, help=None):
    """A KPI card: bordered container with a metric and an optional one-line note."""
    with col.container(border=True):
        st.metric(label, value, delta=delta, help=help)
        if note:
            st.caption(note)


def caveat_line(bt) -> str:
    return (
        f"Backtest {fmt_date(bt['start_date'])} → {fmt_date(bt['end_date'])}, {bt['n_rebalances']} rebalance dates, "
        "signals traded at the next close. It runs **without sentiment** (no point-in-time news), so it tests "
        "the *Price-only rank*, not today's full ranking. The universe has **survivorship bias**, which the "
        "equal-weight universe line shares; the p-value is **gross of costs**."
    )


def require_results():
    """The last run's results, or render the empty state and return None."""
    results = st.session_state.results
    if results is None:
        render_empty_state()
    return results


def render_empty_state():
    st.markdown("#### Screen the S&P 500 in three steps")
    steps = [
        (":material/public:", "Pick a universe",
         "The largest liquid S&P 500 names by market cap. Set the size in the sidebar."),
        (":material/functions:", "Score four factors",
         "Momentum, volume, low volatility and news sentiment, z-scored each day and blended with your weights."),
        (":material/monitoring:", "Backtest the ranking",
         "Hold the top names, rebalance weekly, pay costs, and compare with the universe and SPY, plus a test "
         "for luck."),
    ]
    for col, (icon, title, body) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(body)
    st.button("Run demo (mock sentiment, fast)", type="primary", icon=":material/play_arrow:",
              on_click=request_demo_run, key="demo_run")
    st.caption(
        "Runs the default settings with seeded mock sentiment. Prices still come from Yahoo Finance, so a cold "
        "start takes about a minute; repeat runs are cached for 5 minutes. Or adjust the settings in the "
        "sidebar and click **Run screener**."
    )


# --- Overview ----------------------------------------------------------------
def render_overview():
    results = require_results()
    if results is None:
        return
    latest = latest_scores(results)
    ranked = ranked_scores(latest)
    problem = backtest_problem(results)
    bt = None if problem else results["backtest"]

    cols = st.columns(4)
    kpi(cols[0], "Stocks screened", f"{len(ranked):,}",
        note=f"of {len(latest):,} with prices on {fmt_date(results['as_of'])}",
        help="Names with every weighted factor available on the latest date. The rest are mostly still in "
             "factor warm-up or missing data.")
    if len(ranked):
        top = ranked.index[0]
        kpi(cols[1], "Top pick", top, note=f"Composite {fmt_z(ranked['composite'].iloc[0])} · #1 of {len(ranked)}",
            help="Highest composite score on the latest date.")
    else:
        kpi(cols[1], "Top pick", "n/a", note="No stock has a composite yet")
    if bt:
        verdict, color = p_value_reading(bt["p_value"])
        kpi(cols[2], "Significance (p-value)", fmt_p(bt["p_value"]),
            note=f":{color}-badge[{verdict}] gross of costs",
            help="Roughly the share of random portfolios from the same eligible names that did at least as "
                 "well as the strategy's picks. The most honest number here: it compares like with like, so "
                 "the universe's biases cancel out. See Backtest for how to read it.")
        kpi(cols[3], "Strategy vs universe", fmt_return(bt["excess_vs_universe"]),
            note=f"Strategy {fmt_return(bt['strategy_return'])} · Universe EW "
                 f"{fmt_return(bt['universe_return'])} · SPY {fmt_return(bt['benchmark_return'])}",
            help="The strategy's total return after costs minus an equal-weight portfolio of every eligible "
                 "name (gross of costs), over the backtest window. Beating SPY says little: today's largest "
                 "companies are, by construction, past winners.")
    else:
        kpi(cols[2], "Significance (p-value)", "n/a", note="See the note below")
        kpi(cols[3], "Strategy vs universe", "n/a", note="See the note below")

    with st.container(border=True):
        head, link = st.columns([3, 1], vertical_alignment="center")
        head.markdown("**Strategy vs universe and SPY** · growth of 100")
        if link.button("Full backtest", icon=":material/arrow_forward:", type="tertiary", key="ov_backtest"):
            open_page(PAGE_BACKTEST)
        if bt:
            show_chart(equity_figure(bt, drawdown=False, height=260), key="ov_equity")
            st.caption(caveat_line(bt))
        else:
            st.info(problem)

    head, link = st.columns([3, 1], vertical_alignment="center")
    head.markdown("#### Top picks")
    if link.button("All rankings", icon=":material/arrow_forward:", type="tertiary", key="ov_rankings"):
        open_page(PAGE_RANKINGS)
    with_sentiment = tickers_with_sentiment(results)
    top5 = ranked.head(5)
    for col, (rank, (ticker, row)) in zip(st.columns(5), enumerate(top5.iterrows(), start=1)):
        with col.container(border=True):
            st.markdown(f"**#{rank} · {ticker}**")
            st.caption(f"Composite {fmt_z(row['composite'])}")
            st.markdown(mini_factor_bars_html(row, ticker in with_sentiment), unsafe_allow_html=True)
            if st.button(f"{ticker} details", key=f"ov_open_{ticker}", icon=":material/arrow_forward:",
                         width="stretch", help=f"Open {ticker} in Stock detail"):
                open_page(PAGE_DETAIL, ticker)
    if any(t not in with_sentiment for t in top5.index):
        st.caption("Bars show factor z-scores (±3 = full bar). n/a = no sentiment for that name "
                   "(not fetched, or no headlines found).")


# --- Rankings ----------------------------------------------------------------
TOP_N_OPTIONS = ["10", "20", "50", "All"]


def _restore_top_n():
    if st.session_state.rank_top_n is None:
        st.session_state.rank_top_n = TOP_N_OPTIONS[0]


def rankings_table(results, ranked: pd.DataFrame) -> pd.DataFrame:
    """Display/CSV frame: one row per ranked name; sentiment blank where none was fetched."""
    with_sentiment = tickers_with_sentiment(results)
    table = pd.DataFrame({
        "Rank": range(1, len(ranked) + 1),
        "Price-only rank": ranked["composite_pit"].rank(ascending=False, method="min").to_numpy(),
        "Ticker": ranked.index.astype(str),
        "Composite": ranked["composite"].to_numpy(),
    })
    for c in FACTOR_COLS:
        values = ranked[c]
        if c == "sentiment":
            values = values.where(ranked.index.isin(list(with_sentiment)))
        table[FACTOR_LABELS[c]] = values.to_numpy()
    return table


def render_rankings():
    results = require_results()
    if results is None:
        return
    latest = latest_scores(results)
    ranked = ranked_scores(latest)
    ss = st.session_state

    search_col, top_col = st.columns([3, 2], vertical_alignment="bottom")
    query = search_col.text_input("Search ticker", key="rank_search", placeholder="e.g. AAPL").strip().upper()
    top_col.segmented_control("Show top", TOP_N_OPTIONS, key="rank_top_n", on_change=_restore_top_n)
    top_n = ss.rank_top_n or TOP_N_OPTIONS[0]

    table = rankings_table(results, ranked)
    if query:
        view = table[table["Ticker"].str.contains(query, regex=False)]
    else:
        view = table if top_n == "All" else table.head(int(top_n))
    view = view.reset_index(drop=True)

    z_cols = [FACTOR_LABELS[c] for c in FACTOR_COLS]
    styled = view.style.map(z_cell_css, subset=z_cols).format("{:+.2f}", subset=z_cols, na_rep="")
    cmin, cmax = float(table["Composite"].min()), float(table["Composite"].max())
    if cmin == cmax:
        cmax = cmin + 1e-9
    mock = results["params"]["use_mock"]
    column_config = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "Price-only rank": st.column_config.NumberColumn("Price-only rank", format="%d", width="small",
                                                         help=PIT_RANK_HELP),
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Composite": st.column_config.ProgressColumn(
            "Composite", format="%+.2f", min_value=cmin, max_value=cmax,
            help="Your weighted blend of the four factor z-scores. Bar length is relative to this universe."),
        "Sentiment": st.column_config.NumberColumn(
            "Sentiment", help=SENTIMENT_HELP_MOCK if mock else SENTIMENT_HELP_LIVE),
        **{FACTOR_LABELS[c]: st.column_config.NumberColumn(FACTOR_LABELS[c], help=h) for c, h in FACTOR_HELP.items()},
    }
    if table.empty:
        st.info("No stock has a composite yet: every name is still in factor warm-up.")
    elif view.empty:
        st.info(f"No ticker matches “{query}”.")
    else:
        event = st.dataframe(
            styled, hide_index=True, column_config=column_config, on_select="rerun",
            selection_mode="single-row", key=f"rank_table_{ss.run_id}_{top_n}_{query}",
            height=min(len(view) + 1, 16) * 35 + 3,
        )
        rows = event.selection.rows if event is not None else []
        if rows and rows[0] < len(view):
            ss.selected_ticker = view.loc[rows[0], "Ticker"]

    actions = st.columns([1, 1, 2], vertical_alignment="center")
    csv = view.rename(columns={c: f"{c} z" for c in z_cols}).to_csv(index=False, na_rep="")
    actions[0].download_button("Download CSV", csv.encode("utf-8"), file_name="rankings.csv", mime="text/csv",
                               icon=":material/download:", width="stretch", on_click="ignore")
    selected = ss.selected_ticker
    if selected in set(view["Ticker"]):
        if actions[1].button(f"Open {selected} in Stock detail", icon=":material/arrow_forward:",
                             type="primary", width="stretch", key="rank_open"):
            open_page(PAGE_DETAIL, selected)
    else:
        actions[1].caption("Select a row to open it in Stock detail.")
    unranked = len(latest) - len(ranked)
    if unranked:
        actions[2].caption(f"{unranked} downloaded name(s) are still in factor warm-up and not ranked.")

    st.markdown("#### Factor diagnostics")
    left, right = st.columns(2)
    with left.container(border=True):
        highlight = table["Ticker"] if top_n == "All" else table["Ticker"].head(int(top_n))
        st.markdown("**Composite distribution**" + ("" if top_n == "All" else f" · top {top_n} highlighted"))
        composite = pd.Series(table["Composite"].to_numpy(), index=table["Ticker"])
        show_chart(composite_histogram(composite, pd.Index(highlight) if top_n != "All" else pd.Index([])),
                   key="rank_hist")
    with right.container(border=True):
        st.markdown("**Factor correlation** · ranked names, latest date")
        # Correlate real values only: blank sentiment (z filled with 0) would pull every pair toward 0.
        factors = ranked[list(FACTOR_COLS)].copy()
        factors.loc[~factors.index.isin(list(tickers_with_sentiment(results))), "sentiment"] = float("nan")
        show_chart(correlation_heatmap(factors.corr()), key="rank_corr")
        st.caption("Sentiment pairs use only names with sentiment.")


# --- Backtest ----------------------------------------------------------------
def render_backtest():
    results = require_results()
    if results is None:
        return
    bt = results["backtest"]
    cfg = get_settings().backtest
    problem = backtest_problem(results)
    if problem:
        st.warning(problem, icon=":material/warning:")
        how_to_read(bt or None, cfg)
        return
    if bt["benchmark_series"].nunique() == 1:
        st.warning("SPY prices were unavailable, so the benchmark is flat: SPY return, excess return and hit rate "
                   "are not meaningful for this run.", icon=":material/warning:")

    cols = st.columns(4)
    kpi(cols[0], "Strategy return", fmt_return(bt["strategy_return"]),
        help=f"Total return on {md_usd(bt['initial_capital'])} of starting capital, after transaction costs.")
    kpi(cols[1], "vs equal-weight universe", fmt_return(bt["excess_vs_universe"]),
        note=f"Universe {fmt_return(bt['universe_return'])}",
        help="Strategy return minus an equal-weight portfolio of every eligible name, rebalanced on the same "
             "dates (gross of costs). This is the fair yardstick: it shares the universe's survivorship and "
             "size biases, so the difference isolates the stock selection.")
    kpi(cols[2], "vs SPY", fmt_return(bt["excess_return"]),
        note=f"SPY {fmt_return(bt['benchmark_return'])}",
        help="Strategy return minus SPY (dividend-adjusted), simple difference, not annualized. Flattered by "
             "the universe: today's largest companies are past winners.")
    sharpe = bt["sharpe_ratio"]
    kpi(cols[3], "Sharpe ratio", "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}",
        note=f"Risk-free {bt['risk_free_rate']:.2%} ({bt['risk_free_source']})",
        help="Annualized: mean of (daily return − daily risk-free return) divided by its standard deviation, "
             "times √252. The risk-free rate is the 13-week T-bill yield on each day unless "
             "BACKTEST_RISK_FREE_RATE fixes it.")

    with st.container(border=True, key="secondary_stats"):
        stats = [
            ("Max drawdown", fmt_return(bt["max_drawdown"]) if bt["max_drawdown"] < 0 else "0.00%",
             "Worst peak-to-trough fall of the strategy's value, measured from the second trading day. The chart "
             "below also counts day one as a possible peak, so its low can be slightly deeper."),
            ("Volatility (ann.)", fmt_share(bt["volatility"], 2),
             "Standard deviation of daily strategy returns, times √252."),
            ("Hit rate", fmt_share(bt["hit_rate"]), "Share of trading days on which the strategy's daily return "
                                                    "beat SPY's."),
            ("Turnover", f"{bt['turnover']:.2f}×", "Total traded value (every buy and sell, including the first "
                                                   "purchase) divided by the starting capital."),
            ("Costs paid", fmt_usd(bt["total_costs"]),
             f"{cfg.transaction_cost_bps:g} bps charged on the value of every buy and sell."),
        ]
        for col, (label, value, help) in zip(st.columns(5), stats):
            col.metric(label, value, help=help)

    p = bt["p_value"]
    verdict, color = p_value_reading(p)
    with st.container(border=True):
        st.markdown(f"**:material/casino: Significance: p = {fmt_p(p)}** &nbsp; :{color}-badge[{verdict}]")
        st.markdown(
            f"The p-value is roughly the share of {cfg.n_bootstrap:,} random portfolios (same size, drawn from the "
            f"same eligible names at each of the {bt['n_rebalances']} rebalance dates) that did at least as well as "
            "the strategy's picks, with a +1 correction so it is never exactly 0. It tests the stock selection "
            "within this universe, not whether the strategy beat SPY. It is **gross of costs**, and a few dozen "
            "rebalances is limited evidence."
        )
        st.caption("Reading: below 0.05 unlikely to be luck · 0.05 to 0.20 weak evidence · "
                   "above 0.20 indistinguishable from luck.")

    hold = bt.get("holdout") or {}
    if hold:
        h_verdict, h_color = p_value_reading(hold["p_value"])
        with st.container(border=True):
            st.markdown(f"**:material/lock: Holdout: last {cfg.holdout_fraction:.0%} of the window** "
                        f"({fmt_date(hold['start_date'])} → {fmt_date(bt['end_date'])}, "
                        f"{hold['n_rebalances']} rebalances) &nbsp; :{h_color}-badge[{h_verdict}]")
            h_cols = st.columns(4)
            h_cols[0].metric("Strategy", fmt_return(hold["strategy_return"]))
            h_cols[1].metric("Universe EW", fmt_return(hold["universe_return"]))
            h_cols[2].metric("SPY", fmt_return(hold["benchmark_return"]))
            h_cols[3].metric("p-value", fmt_p(hold["p_value"]))
            st.caption(
                "If you tune the weights, judge them on the earlier part of the window and only then look here. "
                "Weights that hold up in the holdout are less likely to be overfit. Once you start tuning "
                "against the holdout too, it stops being out-of-sample."
            )

    st.markdown("#### Equity curve and drawdown")
    with st.container(border=True):
        show_chart(equity_figure(bt, drawdown=True, height=460), key="bt_equity")
        st.caption(
            f"{fmt_date(bt['start_date'])} → {fmt_date(bt['end_date'])} · {bt['n_rebalances']} rebalance dates "
            f"(every {cfg.rebalance_freq} trading days, including the first purchase) · top {cfg.top_n_stocks} "
            "names, reset to equal weight at every rebalance · "
            f"signals traded at the next close · final value {md_usd(bt['final_value'])} from "
            f"{md_usd(bt['initial_capital'])}."
        )
    how_to_read(bt, cfg)


def how_to_read(bt, cfg):
    window = (f"{fmt_date(bt['start_date'])} → {fmt_date(bt['end_date'])} with {bt['n_rebalances']} rebalance dates"
              if bt else "not available for this run")
    with st.container(border=True):
        st.markdown("**:material/info: How to read this**")
        st.markdown(
            "- **No sentiment in the backtest.** There is no point-in-time news history, so the backtest ranks on "
            "momentum, volume and volatility only. Today's ranking does use sentiment; the **Price-only rank** "
            "column on Rankings shows the ranking the backtest tested.\n"
            "- **Survivorship and size bias.** The universe is today's S&P 500, ranked by today's market cap, so "
            "every name is a company that grew large, and none has since left the index. Any portfolio of them "
            "tends to beat SPY in hindsight. Compare with the **equal-weight universe**, which shares these "
            "biases, and trust the p-value over the gap to SPY.\n"
            "- **Traded at the next close.** Scores use data up to the close of day t; the portfolio trades on them "
            "at the close of day t+1.\n"
            f"- **Costs.** Returns include {cfg.transaction_cost_bps:g} bps per trade; the p-value is gross of "
            "costs.\n"
            f"- **Window.** {window}."
        )
        with st.expander("More on the method and its limits"):
            st.markdown(
                "Factors need history before they can score (momentum needs about 13 months), so about 400 extra "
                "calendar days are downloaded before the window. Each rebalance resets the portfolio to equal "
                "weights, paying costs on every trade that takes. If you tune weights here, check them on the "
                "holdout or a different period: tuning and testing on the same window overfits.\n\n"
                f"Full write-up: [How the backtest works (and what it can't tell you)]({README_BACKTEST_URL})."
            )


# --- Stock detail ------------------------------------------------------------
def render_stock_detail():
    if require_results() is None:
        return
    stock_detail_body()


def _sync_selected_ticker():
    st.session_state.selected_ticker = st.session_state.detail_ticker


@st.fragment
def stock_detail_body():
    """Fragment: changing the ticker reruns only this page body, not the whole app."""
    ss = st.session_state
    results = ss.results
    latest = latest_scores(results)
    ranked = ranked_scores(latest)
    options = list(ranked.index) + [t for t in latest.index if t not in ranked.index]
    if not options:
        st.info("No stocks available.")
        return
    want = ss.selected_ticker if ss.selected_ticker in options else options[0]
    ss.selected_ticker = want
    if ss.get("detail_ticker") != want:
        ss.detail_ticker = want
    rank_of = {t: i for i, t in enumerate(ranked.index, start=1)}
    st.selectbox("Ticker", options, key="detail_ticker", on_change=_sync_selected_ticker,
                 format_func=lambda t: f"{t}  ·  #{rank_of[t]}" if t in rank_of else f"{t}  ·  not ranked",
                 width=320)
    ticker = ss.detail_ticker
    row = latest.loc[ticker]

    st.subheader(ticker)
    cols = st.columns(3)
    if ticker in rank_of:
        pct = round(ranked["composite"].rank(pct=True)[ticker] * 100)
        kpi(cols[0], "Rank", f"#{rank_of[ticker]} of {len(ranked)}")
        kpi(cols[1], "Composite", fmt_z(row["composite"]), help="Weighted blend of the factor z-scores.")
        kpi(cols[2], "Percentile", ordinal(pct), help="Share of ranked names with a composite at or below this one.")
    else:
        kpi(cols[0], "Rank", "not ranked", note="Still in factor warm-up")

    with_sentiment = tickers_with_sentiment(results)
    has_sentiment = ticker in with_sentiment
    values = {c: (row[c] if (c != "sentiment" or has_sentiment) else None) for c in FACTOR_COLS}
    medians = {c: float(ranked[c].median()) if len(ranked) else 0.0 for c in FACTOR_COLS}
    covered = ranked.index.isin(list(with_sentiment))
    medians["sentiment"] = float(ranked.loc[covered, "sentiment"].median()) if covered.any() else 0.0

    left, right = st.columns([1, 1])
    with left.container(border=True):
        st.markdown("**Factor z-scores** · bar = this stock, tick = universe median")
        show_chart(factor_bar_figure(values, medians), key="detail_factors")
    with right.container(border=True):
        st.markdown("**Sentiment**")
        source = sentiment_source(results, ticker)
        entry = results["sentiment"].get(ticker, {}) if has_sentiment else {}
        entry = entry if isinstance(entry, dict) else {"score": entry}
        s_cols = st.columns(3)
        s_cols[0].metric("Score", fmt_z(entry.get("score")) if has_sentiment else "n/a",
                         help="Headline sentiment from -1 (negative) to +1 (positive), shrunk toward 0 by "
                              f"n / (n + {get_settings().sentiment.prior_headlines:g}) for n headlines, so a "
                              "single headline can't give an extreme score.")
        if "n_headlines" in entry:
            s_cols[1].metric("Headlines", f"{entry['n_headlines']}",
                             help=f"Headlines scored. Unshrunk score: {fmt_z(entry.get('raw_score'))}.")
        else:
            s_cols[1].metric("Confidence", f"{entry['confidence']:.2f}" if "confidence" in entry else "n/a",
                             help="Agreement between headlines, 0 to 1 (1 minus the spread of headline "
                                  "scores). Not a probability.")
        s_cols[2].metric("Source", source)
        if source == "mock":
            st.caption("Mock sentiment is seeded random data for the demo, unrelated to the headlines below.")
        elif source in ("not fetched", "no headlines"):
            st.caption("No sentiment for this name, so its sentiment z-score is 0 in the composite "
                       "(the average of the names with news).")
        st.markdown("**Recent headlines**")
        try:
            items = stages.headlines(ticker)
        except Exception:  # noqa: BLE001 - headlines are optional context
            items = None
        if items:
            st.markdown("\n".join(headline_markdown(h) for h in items))
            news_cfg = get_settings().sentiment
            st.caption(f"Up to {news_cfg.max_headlines} headlines from the last {news_cfg.lookback_days} days: "
                       "NewsAPI matches for the company name in the title if a key is set, otherwise Yahoo "
                       "Finance's news for the ticker.")
        else:
            st.caption("No recent headlines found." if items is not None else "Couldn't load headlines right now.")

    prices = results["prices"].xs(ticker, level="Ticker")
    prices.index = naive_index(prices.index)
    window = prices.loc[prices.index >= pd.Timestamp(results["params"]["start_date"])]
    with st.container(border=True):
        st.markdown(f"**Adjusted price and volume** · {results['params']['start_date']} → "
                    f"{fmt_date(window.index.max())}" if len(window) else "**Adjusted price and volume**")
        if len(window):
            show_chart(price_volume_figure(window), key="detail_price")
        else:
            st.info("No prices in the selected window.")


# --- navigation --------------------------------------------------------------
PAGE_OVERVIEW = st.Page(render_overview, title="Overview", icon=":material/dashboard:", url_path="overview",
                        default=True)
PAGE_RANKINGS = st.Page(render_rankings, title="Rankings", icon=":material/format_list_numbered:",
                        url_path="rankings")
PAGE_BACKTEST = st.Page(render_backtest, title="Backtest", icon=":material/monitoring:", url_path="backtest")
PAGE_DETAIL = st.Page(render_stock_detail, title="Stock detail", icon=":material/insights:",
                      url_path="stock-detail")
PAGES = [PAGE_OVERVIEW, PAGE_RANKINGS, PAGE_BACKTEST, PAGE_DETAIL]
