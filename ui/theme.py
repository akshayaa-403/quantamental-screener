"""Visual system: color roles, number formats and shared chart helpers.

Colors are mid-tones chosen to read on both the light and the dark theme.
Gain/loss is never shown by color alone: every signed value carries a +/- sign.
"""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.state import FACTOR_COLS

# --- color roles -------------------------------------------------------------
ACCENT = "#6366f1"     # brand / interactive (indigo); also the strategy line
BENCHMARK = "#94a3b8"  # SPY and "rest of universe" (slate)
LOSS = "#dc2626"       # signed losses only (drawdowns); gains use st.metric's own green delta
# One colorblind-safe diverging scale (orange <-> blue) for z-scores and correlations;
# .streamlit/config.toml's chartDivergingColors uses the same endpoints.
Z_NEG_RGB = (234, 88, 12)   # #ea580c
Z_POS_RGB = (37, 99, 235)   # #2563eb
Z_NEG = "rgb({}, {}, {})".format(*Z_NEG_RGB)
Z_POS = "rgb({}, {}, {})".format(*Z_POS_RGB)
# Opaque neutral midpoint so Plotly's automatic cell-text contrast works in both themes.
Z_COLORSCALE = [[0.0, Z_NEG], [0.5, "rgb(148, 163, 184)"], [1.0, Z_POS]]
# Identity colors for the weights bar only (each segment is also labelled in text).
FACTOR_COLORS = {"momentum": "#6366f1", "sentiment": "#0ea5e9", "volume": "#f59e0b", "volatility": "#ec4899"}
FACTOR_LABELS = {c: c.capitalize() for c in FACTOR_COLS}


# --- number formats ----------------------------------------------------------
def _ok(x) -> bool:
    return x is not None and not pd.isna(x)


def fmt_return(x) -> str:
    """Fractional return as a signed percentage: 0.1234 -> '+12.34%'."""
    return f"{x * 100:+.2f}%" if _ok(x) else "n/a"


def fmt_share(x, decimals=1) -> str:
    """Unsigned fraction as a percentage (hit rate, volatility, confidence)."""
    return f"{x * 100:.{decimals}f}%" if _ok(x) else "n/a"


def fmt_usd(x) -> str:
    if not _ok(x):
        return "n/a"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def md_usd(x) -> str:
    """fmt_usd for markdown text: two bare $ signs in one string would render as LaTeX."""
    return fmt_usd(x).replace("$", "\\$")


def fmt_z(x) -> str:
    return f"{x:+.2f}" if _ok(x) else "n/a"


def fmt_p(x) -> str:
    if not _ok(x):
        return "n/a"
    return "< 0.001" if x < 0.0005 else f"{x:.3f}"


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def p_value_reading(p):
    """(verdict, badge color) for the selection p-value, judged on the shown 3-decimal value."""
    if not _ok(p):
        return "not available", "gray"
    p = round(p, 3)
    if p < 0.05:
        return "unlikely to be luck", "blue"
    if p <= 0.20:
        return "weak evidence", "orange"
    return "indistinguishable from luck", "violet"


# --- tables ------------------------------------------------------------------
def z_cell_css(v) -> str:
    """Styler.map color for a z-score cell: diverging, centered at 0, clipped at +/-3.

    The tint is translucent so the theme's own text color stays readable on it.
    """
    if not _ok(v):
        return ""
    r, g, b = Z_POS_RGB if v >= 0 else Z_NEG_RGB
    alpha = min(abs(float(v)), 3.0) / 3.0 * 0.55
    return f"background-color: rgba({r}, {g}, {b}, {alpha:.3f})"


# --- charts ------------------------------------------------------------------
def style_fig(fig, height=320, legend=True):
    """The shared layout: tight margins, horizontal legend on top, unified hover."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 8, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def show_chart(fig, key=None):
    st.plotly_chart(fig, theme="streamlit", width="stretch", key=key, config={"displayModeBar": False})


def naive_index(index: pd.Index) -> pd.Index:
    return index.tz_localize(None) if getattr(index, "tz", None) is not None else index


def equity_figure(bt: dict, drawdown: bool, height: int):
    """Strategy vs SPY, start = 100, optionally with an underwater chart below."""
    port, bench = bt["portfolio_series"], bt["benchmark_series"]
    # Normalize off the TRUE starting capital so the curve's endpoint matches
    # the reported Strategy/SPY return metrics exactly.
    base = bt.get("initial_capital", port.iloc[0])
    strat, spy = port / base * 100, bench / base * 100
    rows = 2 if drawdown else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.68, 0.32] if drawdown else None)
    fig.add_trace(go.Scatter(x=strat.index, y=strat, name="Strategy", line=dict(color=ACCENT, width=2),
                             hovertemplate="%{y:.2f}<extra>Strategy</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY", line=dict(color=BENCHMARK, width=1.6),
                             hovertemplate="%{y:.2f}<extra>SPY</extra>"), row=1, col=1)
    fig.add_hline(y=100, line_dash="dot", line_color=BENCHMARK, opacity=0.6, row=1, col=1)
    fig.update_yaxes(title_text="Growth of 100", row=1, col=1)
    if drawdown:
        dd_strat = port / port.cummax() - 1
        dd_spy = bench / bench.cummax() - 1
        fig.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat, name="Strategy drawdown", fill="tozeroy",
                                 line=dict(color=LOSS, width=1), fillcolor="rgba(220, 38, 38, 0.25)",
                                 hovertemplate="%{y:.2%}<extra>Strategy drawdown</extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY drawdown",
                                 line=dict(color=BENCHMARK, width=1, dash="dot"),
                                 hovertemplate="%{y:.2%}<extra>SPY drawdown</extra>"), row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(hoverformat="%Y-%m-%d")
    fig.update_layout(hovermode="x unified")
    return style_fig(fig, height=height)


def composite_histogram(composite: pd.Series, top: pd.Index):
    """Composite distribution with the top-N slice highlighted (same bins for both)."""
    lo, hi = float(composite.min()), float(composite.max())
    size = (hi - lo) / 20 if hi > lo else 1.0
    bins = dict(start=lo, end=hi + size * 1e-6, size=size)
    rest = composite.drop(top, errors="ignore")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rest, xbins=bins, name="Rest of universe", marker_color=BENCHMARK,
                               hovertemplate="Composite %{x}<br>%{y} stocks<extra>Rest</extra>"))
    if len(top) and len(rest):
        fig.add_trace(go.Histogram(x=composite.loc[top], xbins=bins, name=f"Top {len(top)}",
                                   marker_color=ACCENT,
                                   hovertemplate="Composite %{x}<br>%{y} stocks<extra>Top</extra>"))
    fig.update_layout(barmode="stack", bargap=0.06)
    fig.update_xaxes(title_text="Composite score", tickformat="+.1f")
    fig.update_yaxes(title_text="Stocks")
    return style_fig(fig, height=300)


def correlation_heatmap(corr: pd.DataFrame):
    labels = [FACTOR_LABELS.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale=Z_COLORSCALE,
        texttemplate="%{z:+.2f}", hovertemplate="%{y} vs %{x}: %{z:+.2f}<extra></extra>",
        colorbar=dict(thickness=10, tickformat="+.1f"),
    ))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=300, legend=False)


def factor_bar_figure(values: dict, medians: dict):
    """This stock's factor z-scores as diverging bars, with the universe median marked.

    ``values`` maps factor -> z-score, or None when the factor is unavailable.
    """
    labels = [FACTOR_LABELS[c] for c in values]
    x = [0.0 if v is None else float(v) for v in values.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=labels, orientation="h", name="This stock",
        marker_color=[Z_POS if v >= 0 else Z_NEG for v in x],
        text=["no sentiment" if v is None else fmt_z(v) for v in values.values()],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f}<extra>This stock</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[medians[c] for c in values], y=labels, mode="markers", name="Universe median",
        marker=dict(symbol="line-ns", size=22, line=dict(width=3, color=BENCHMARK)),
        hovertemplate="%{y} median: %{x:+.2f}<extra></extra>",
    ))
    fig.update_xaxes(range=[-3.6, 3.6], zeroline=True, zerolinecolor=BENCHMARK, tickformat="+.0f",
                     title_text="z-score (0 = universe average)")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=260)


def price_volume_figure(prices: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Adj. close", line=dict(color=ACCENT, width=2),
                             hovertemplate="$%{y:,.2f}<extra>Adj. close</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=prices.index, y=prices["Volume"], name="Volume", marker_color=BENCHMARK,
                         hovertemplate="%{y:,.0f}<extra>Volume</extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Adjusted price", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)
    fig.update_xaxes(hoverformat="%Y-%m-%d")
    fig.update_layout(hovermode="x unified")
    return style_fig(fig, height=420)


# --- small HTML pieces (inline styles only, theme-neutral colors) -------------
def weights_bar_html(weights: dict) -> str:
    """One thin stacked bar of the normalized weights, with a text legend."""
    parts = [(FACTOR_LABELS[c], FACTOR_COLORS[c], w) for c, w in weights.items()]
    segments = "".join(
        f'<div style="flex:{w:.6f};background:{color}" title="{label} {w:.0%}"></div>'
        for label, color, w in parts if w > 0
    )
    legend = " ".join(
        f'<span style="white-space:nowrap;margin-right:.6rem">'
        f'<span style="color:{color}">&#9632;</span> {label} {w:.0%}</span>'
        for label, color, w in parts
    )
    summary = ", ".join(f"{label} {w:.0%}" for label, _, w in parts)
    return (
        f'<div role="img" aria-label="Normalized weights: {summary}" '
        f'style="display:flex;gap:2px;height:10px;border-radius:5px;overflow:hidden">{segments}</div>'
        f'<div style="font-size:0.8rem;opacity:0.85;margin-top:0.4rem;line-height:1.7">{legend}</div>'
    )


def mini_factor_bars_html(row: pd.Series, sentiment_ok: bool) -> str:
    """Compact diverging bar per factor for a pick card (bar length = |z| / 3)."""
    lines = []
    for c in FACTOR_COLS:
        v = row[c]
        missing = (c == "sentiment" and not sentiment_ok) or not _ok(v)
        if missing:
            bar, text = "", '<span title="no sentiment" style="opacity:.6">n/a</span>'
        else:
            width = min(abs(float(v)), 3.0) / 3.0 * 50
            left = 50 if v >= 0 else 50 - width
            color = Z_POS if v >= 0 else Z_NEG
            bar = (f'<div style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;top:0;bottom:0;'
                   f'background:{color};border-radius:3px"></div>')
            text = fmt_z(v)
        lines.append(
            '<div style="font-size:.78rem;line-height:1.35;margin-bottom:.3rem">'
            '<div style="display:flex;justify-content:space-between;gap:.5rem">'
            f'<span style="opacity:.75">{FACTOR_LABELS[c]}</span>'
            f'<span style="font-variant-numeric:tabular-nums">{text}</span></div>'
            '<div style="position:relative;height:7px;margin-top:2px;background:rgba(127,127,127,.16);'
            'border-radius:3px"><div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;'
            f'background:rgba(127,127,127,.6)"></div>{bar}</div></div>'
        )
    return "".join(lines)


def escape(text) -> str:
    return html.escape(str(text))
