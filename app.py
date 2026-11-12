"""Signal Desk - local web dashboard (Streamlit). One place for both indicator
tools instead of separate front ends per tool: a real trader doesn't run five
different dashboards to check five related numbers.

    uv run streamlit run app.py

Tabs:
  Dark Pool - Massive live/delayed trade feed (dark_pool/live.py), filtered to
              off-exchange (FINRA TRF) prints, ranked by today's cross-sectional
              outlier z-score (dark_pool.live.todays_outliers)
  FINRA      - heaviest off-exchange (ATS) volume by symbol, free weekly data
  Analysis   - one symbol's participation-rate history, reversal/follow-through
               test, and a prediction for the current week (dark_pool/analysis.py)
  NOI        - NYSE order imbalance auction events, read directly from the NOI
               tool's own data folder (sibling project, no cross-repo import -
               just reads its CSV/JSON output the same way its own dashboard does)
  After-Market Event Analysis - abnormal return around a known event date (earnings, an outage,
               guidance...) vs a benchmark, with a market-model t-stat reliability
               gate (event_study/study.py), plus a cross-event signature and a
               live-case matcher built on a growing event library (event_study/event_library.py)
  Anomaly Detector - today's trade-count outliers across the S&P 500 (z-score vs
               each stock's own trailing 5-day baseline), following Massive's
               "Hunting Anomalies in the Stock Market" tutorial (anomaly/detector.py)
  Insider Indicator - COIN-only Form 3/4/13F ownership read vs price (sibling
               insider-indicator-tool's indicator.py, imported directly - see the
               sys.path note below for why that's fine here but not for NOI/dark_pool)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dark_pool.analysis import MIN_ROWS_FOR_TEST, participation_series, predict_this_week, reversal_test
from event_study.study import DEFAULT_BASELINE, DEFAULT_ESTIMATION_DAYS, DEFAULT_WINDOW, event_study
from event_study.event_library import EVENT_TYPES, add_event, event_sample_study, load_events, match_new_case
from dark_pool.finra.collect import FINRA_DIR
from dark_pool.live import available_days as live_available_days, todays_outliers
from dark_pool.scanner import available_weeks, top_symbols
from anomaly.detector import DEFAULT_THRESHOLD as ANOMALY_DEFAULT_THRESHOLD, scan as anomaly_scan

# insider-indicator-tool is a separate sibling repo/project, with its own git
# history and its own Flask/Postgres-only dependencies (Flask, psycopg) that this
# app doesn't need and shouldn't require installing. Unlike NOI/dark_pool (which
# this app reads as plain output files, never importing their code, specifically
# to avoid depending on their .venv), indicator.py itself only imports pandas,
# the massive SDK and python-dotenv - all already in THIS project's own venv - so
# importing it directly (instead of re-implementing its analytics here) is safe:
# no extra install, no dependency on insider-indicator-tool's own .venv/Flask/DB.
INSIDER_ROOT = Path(__file__).resolve().parent.parent / "insider-indicator-tool"
if str(INSIDER_ROOT) not in sys.path:
    sys.path.insert(0, str(INSIDER_ROOT))
import indicator as insider_indicator  # noqa: E402 (path must be set up first)
# Same cache folder indicator.py's own CLI defaults to, so filings cached from
# here are reused by (and reuse) runs of insider-indicator-tool's own CLI/Flask app.
INSIDER_CACHE_DIR = INSIDER_ROOT / ".insider_cache"

CONSTITUENTS_PATH = FINRA_DIR.parent.parent / "reference_data" / "sp500_constituents.csv"

# Same reference palette used across the NOI tool and market-charts, for a
# consistent look across all three projects.
UP, DOWN, BLUE, NEUTRAL = "#0ca30c", "#d03b3b", "#2a78d6", "#898781"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

st.set_page_config(page_title="Signal Desk", layout="wide")

# Streamlit applies its own dark theme on top of plotly figures by default, which
# overrides these colors (title/axis-title text goes near-white, invisible against
# the light paper_bgcolor below) - every st.plotly_chart call must pass theme=None
# to actually get these colors instead of Streamlit's.
PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=INK, size=13),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK, size=13),
                title=dict(font=dict(color=INK, size=14))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK, size=13),
                title=dict(font=dict(color=INK, size=14))),
    margin=dict(t=60, l=10, r=10, b=10),
)


def plot_title(text: str) -> dict:
    """A centered, legible chart title - call this for the title= kwarg on every
    chart (not baked into PLOTLY_LAYOUT, since every chart passes its own title=
    text alongside **PLOTLY_LAYOUT in the same update_layout call)."""
    return dict(text=text, x=0.5, xanchor="center", y=0.97, yanchor="top",
                font=dict(color=INK, size=18))


def ats_info_box():
    with st.expander("ℹ️ What is ATS / FINRA TRF data?"):
        st.markdown(
            "**ATS (Alternative Trading System)** is the formal name for what's commonly "
            "called a **dark pool** - a private venue where large trades are matched away "
            "from public exchanges like the NYSE or Nasdaq, without showing the order "
            "beforehand. Every ATS trade must still be reported to FINRA through one of "
            "three **TRFs** (Trade Reporting Facilities): 201 (NYSE TRF), 202 (NASDAQ TRF "
            "Carteret), or 203 (NASDAQ TRF Chicago) - that's where this data comes from.\n\n"
            "**Shares** = total volume that traded off-exchange. **Trades** = number of "
            "individual fills (not shares) - a high share count from few trades usually "
            "means a small number of large block trades; from many trades usually means "
            "many smaller fills."
        )

DISPLAY_COLUMNS = {
    "symbol": "Symbol",
    "ats_shares": "ATS Shares (Alternative Trading System)",
    "ats_trades": "ATS Trades (Alternative Trading System)",
}

COLUMN_CONFIG = {
    "Symbol": st.column_config.TextColumn(
        "Symbol", help="The stock's ticker symbol.",
    ),
    "ATS Shares (Alternative Trading System)": st.column_config.NumberColumn(
        "ATS Shares (Alternative Trading System)", format="localized",
        help=(
            "An Alternative Trading System (ATS) is a private venue - a 'dark pool' - "
            "where large trades are matched away from public exchanges like the NYSE "
            "or Nasdaq, without showing the order beforehand. This is the total number "
            "of shares in this stock that traded through ATS venues this week, summed "
            "across every ATS operator that reported to FINRA."
        ),
    ),
    "ATS Trades (Alternative Trading System)": st.column_config.NumberColumn(
        "ATS Trades (Alternative Trading System)", format="localized",
        help=(
            "The number of individual trades (not shares) executed through ATS/dark "
            "pool venues this week. A high share count from few trades usually means "
            "a small number of large block trades; a high share count from many trades "
            "usually means many smaller fills."
        ),
    ),
}

# NOI tool's own data folder - sibling project on disk. Read directly (CSV/JSON
# only), never imported as a package, so this app has no dependency on the NOI
# tool's own .venv or code.
NOI_ROOT = Path(__file__).resolve().parent.parent / "NOI tool (order imbalance)"
NOI_DATA_DIR = NOI_ROOT / "data"
NOI_STATUS_PATH = NOI_DATA_DIR / "collector_status.json"


@st.cache_data
def load_constituents() -> pd.DataFrame:
    if not CONSTITUENTS_PATH.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector", "sub_industry"])
    return pd.read_csv(CONSTITUENTS_PATH)


def noi_available_days() -> list[str]:
    noi_dir = NOI_DATA_DIR / "noi"
    if not noi_dir.exists():
        return []
    return sorted(p.stem for p in noi_dir.glob("*.csv"))


st.title("Signal Desk")
st.caption("Two off-exchange signal tools, one dashboard: dark pool (FINRA TRF) prints and NYSE order imbalance auctions.")

tab_dark, tab_finra, tab_analysis, tab_noi, tab_event, tab_anomaly, tab_insider = st.tabs(
    ["Dark Pool (Live)", "FINRA (Free, Weekly)", "Analysis", "NOI (Order Imbalance)",
     "After-Market Event Analysis", "Anomaly Detector", "Insider Indicator"]
)

# ---------------------------------------------------------------------------
with tab_dark:
    st.caption("Massive's trade feed, filtered to off-exchange (FINRA TRF) prints. "
               "15-min delayed on the Stocks Developer plan. Run: `uv run python -m dark_pool.live`")
    days = live_available_days()
    if not days:
        st.warning("No live data yet. Start the collector: `uv run python -m dark_pool.live`")
    else:
        day = st.selectbox("Day", options=list(reversed(days)), index=0)
        result = todays_outliers(day)
        if result.empty:
            st.info("No off-exchange prints logged for this day yet (min-notional threshold, or market closed).")
        else:
            top = result.head(25)
            colors = [DOWN if o else BLUE for o in top["outlier"]]
            fig = go.Figure(go.Bar(
                x=top["participation_pct_of_adv"][::-1], y=top["symbol"][::-1], orientation="h",
                marker_color=colors[::-1],
                hovertemplate="%{y}: %{x:.2f}%% of ADV today<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"Today's dark-pool participation (% of 20-day ADV) - {day}"),
                               height=max(320, 24 * len(top)),
                               xaxis_title="% of 20-day average daily volume (ADV) traded off-exchange today",
                               yaxis_title="Symbol")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.caption("Red bars: |z-score| >= 2 against today's other symbols - unusually heavy dark flow right now, not just a big number.")
            st.dataframe(
                result[["symbol", "cum_shares_today", "adv20", "participation_pct_of_adv", "z_score", "outlier"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "symbol": "Symbol",
                    "cum_shares_today": st.column_config.NumberColumn("Dark Shares Today", format="localized"),
                    "adv20": st.column_config.NumberColumn("20-Day Avg Daily Volume", format="localized"),
                    "participation_pct_of_adv": st.column_config.NumberColumn("% of ADV Today", format="%.2f%%"),
                    "z_score": st.column_config.NumberColumn("Z-Score (vs. today's other symbols)", format="%.2f"),
                    "outlier": st.column_config.CheckboxColumn("Outlier (|z| >= 2)"),
                },
            )
            ats_info_box()

# ---------------------------------------------------------------------------
with tab_finra:
    weeks = available_weeks()
    if not weeks:
        st.warning("No data pulled yet. Run: `uv run python -m dark_pool.finra.collect --all`")
    else:
        col1, col2 = st.columns([2, 1])
        week = col1.selectbox("Week", options=list(reversed(weeks)), index=0)
        top_n = col2.slider("Top N", 5, 50, 20)

        result = top_symbols(week, top_n)
        if result.empty:
            st.info("No ATS rows for this week.")
        else:
            fig = go.Figure(go.Bar(
                x=result["symbol"], y=result["ats_shares"],
                marker_color=BLUE, hovertemplate="%{x}: %{y:,.0f} shares<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"Heaviest off-exchange (ATS / dark pool) volume - week of {week}"),
                               height=500,
                               xaxis_title="Symbol", yaxis_title="ATS (dark pool) shares traded that week")
            # Angled ticker labels need real bottom margin, and a rotated y-axis title
            # needs real left margin too, or the tick numbers get squeezed out entirely
            # (PLOTLY_LAYOUT's defaults, b=10/l=10, are both far too tight here).
            fig.update_layout(margin=dict(t=60, l=70, r=10, b=120))
            # Category labels (tickers) need their own explicit font + angle, or they
            # get crushed illegibly small once there are 20-50 of them side by side.
            fig.update_xaxes(type="category", tickangle=-45, tickfont=dict(color=INK, size=13))
            fig.update_yaxes(tickformat="~s", tickfont=dict(color=INK, size=13))  # compact: 100M not 100,000,000
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.dataframe(result.rename(columns=DISPLAY_COLUMNS), use_container_width=True, hide_index=True,
                         column_config=COLUMN_CONFIG)
            ats_info_box()

# ---------------------------------------------------------------------------
with tab_analysis:
    constituents = load_constituents()
    if constituents.empty:
        st.error(f"No S&P 500 reference data found at {CONSTITUENTS_PATH}.")
        symbol = None
    else:
        sectors = sorted(constituents["sector"].dropna().unique())
        col1, col2 = st.columns(2)
        default_sector_ix = sectors.index("Information Technology") if "Information Technology" in sectors else 0
        sector = col1.selectbox("Sector", options=sectors, index=default_sector_ix)

        in_sector = constituents[constituents["sector"] == sector].sort_values("symbol")
        sub_industries = sorted(in_sector["sub_industry"].dropna().unique())
        sub_industry = col2.selectbox("Industry group", options=["All"] + sub_industries)
        if sub_industry != "All":
            in_sector = in_sector[in_sector["sub_industry"] == sub_industry]

        options = in_sector["symbol"].tolist()
        labels = {row["symbol"]: f"{row['symbol']} - {row['name']}" for _, row in in_sector.iterrows()}
        symbol = st.selectbox("Symbol", options=options, format_func=lambda s: labels.get(s, s)) if options else None

    if symbol:
        try:
            series = participation_series(symbol)
        except SystemExit as e:
            st.error(str(e))
            series = None

        if series is not None and not series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["week"], y=series["participation_pct"], mode="lines+markers",
                                      line=dict(color=BLUE), name="Participation %"))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"{symbol} - off-exchange (ATS / dark pool) share of weekly volume"),
                               height=500,
                               xaxis_title="Week (Monday)", yaxis_title="% of weekly volume traded off-exchange (ATS)")
            # Angled date labels need real bottom margin, and the rotated y-axis title
            # needs real left margin too, or the tick numbers get squeezed out entirely.
            fig.update_layout(margin=dict(t=60, l=60, r=10, b=100))
            # With 50+ weeks of history, date labels crush together illegibly unless
            # the tick count is capped and they're angled instead of flat.
            fig.update_xaxes(tickangle=-45, nticks=15, tickfont=dict(color=INK, size=12))
            fig.update_yaxes(ticksuffix="%", tickfont=dict(color=INK, size=13))
            st.plotly_chart(fig, use_container_width=True, theme=None)
            ats_info_box()

            weeks_with_z = series.dropna(subset=["z_score"])
            st.caption(f"{len(weeks_with_z)} weeks with a z-score computed "
                       f"(needs {MIN_ROWS_FOR_TEST} with a known next-week outcome for a reliable test).")

            st.subheader("This week's prediction")
            pred = predict_this_week(symbol)
            if pred:
                reliable = pred["reliable"]
                badge = "✅ statistically reliable (|t| ≥ 2)" if reliable else "⚠️ not yet reliable (|t| < 2) - treat as exploratory"
                c1, c2, c3 = st.columns(3)
                c1.metric("Participation this week", f"{pred['participation_pct']}%", f"z={pred['z_score']}")
                c2.metric("Predicted return", f"{pred['predicted_return_pct']:+.2f}%")
                c3.metric("Predicted price", f"${pred['predicted_price']}", f"from ${pred['last_close']}")
                st.caption(f"Fit on {pred['n_weeks_fit']} weeks · beta={pred['beta']} · t-stat={pred['t_stat']} · {badge}")
            else:
                st.info("Not enough historical weeks with a known outcome yet - see console/log for the exact count.")

            st.subheader("Reversal / follow-through test (full detail)")
            with st.expander("Show test output"):
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    reversal_test(symbol)
                st.text(buf.getvalue())

# ---------------------------------------------------------------------------
with tab_noi:
    st.caption("NYSE order imbalance auction events (opening/closing). Collector runs in the NOI tool project "
               "(Windows Scheduled Task, weekdays 8:55am-4:10pm ET). This reads its output directly.")

    if NOI_STATUS_PATH.exists():
        status = pd.read_json(NOI_STATUS_PATH, typ="series")
        c1, c2, c3 = st.columns(3)
        c1.metric("Collector status", status.get("status", "unknown"))
        c2.metric("Events logged (this run)", f"{status.get('events', 0):,}")
        c3.metric("Last updated", str(status.get("updated", "-"))[:19])
    else:
        st.info("NOI collector hasn't run yet - no status file found.")

    noi_days = noi_available_days()
    if not noi_days:
        st.warning("No NOI auction data yet. Collector runs weekdays 8:55am-4:10pm ET; check back after the next session.")
    else:
        noi_day = st.selectbox("Day", options=list(reversed(noi_days)), index=0, key="noi_day")
        noi_df = pd.read_csv(NOI_DATA_DIR / "noi" / f"{noi_day}.csv")
        if noi_df.empty:
            st.info("No NOI events logged for this day.")
        else:
            noi_df["abs_imbalance"] = noi_df["imbalance"].abs()
            top_imbalances = noi_df.sort_values("abs_imbalance", ascending=False).head(25)
            fig = go.Figure(go.Bar(
                x=top_imbalances["imbalance"][::-1], y=top_imbalances["symbol"][::-1], orientation="h",
                marker_color=[UP if v > 0 else DOWN for v in top_imbalances["imbalance"][::-1]],
                hovertemplate="%{y}: %{x:,.0f} shares imbalance<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"Largest NOI auction imbalances - {noi_day}"),
                               height=max(320, 24 * len(top_imbalances)),
                               xaxis_title="Imbalance (shares, + = buy-side, - = sell-side)",
                               yaxis_title="Symbol")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.dataframe(
                noi_df[["time_et", "symbol", "auction_type", "imbalance", "paired", "indicative_price"]]
                .sort_values("time_et", ascending=False),
                use_container_width=True, hide_index=True,
            )

# ---------------------------------------------------------------------------
with tab_event:
    st.caption("Was the move around this date actually unusual, or just the market moving that day? "
               "Fits a market-model regression on the prior trading days, then measures Cumulative "
               "Abnormal Return (CAR) in the event window - the part of the move the benchmark doesn't explain.")

    event_constituents = load_constituents()
    if event_constituents.empty:
        st.error(f"No S&P 500 reference data found at {CONSTITUENTS_PATH}.")
        event_symbol = None
    else:
        event_sectors = sorted(event_constituents["sector"].dropna().unique())
        c1, c2 = st.columns(2)
        default_sector_ix = event_sectors.index("Information Technology") if "Information Technology" in event_sectors else 0
        event_sector = c1.selectbox("Sector", options=event_sectors, index=default_sector_ix, key="event_sector")

        event_in_sector = event_constituents[event_constituents["sector"] == event_sector].sort_values("symbol")
        event_sub_industries = sorted(event_in_sector["sub_industry"].dropna().unique())
        event_sub_industry = c2.selectbox("Industry group", options=["All"] + event_sub_industries, key="event_sub_industry")
        if event_sub_industry != "All":
            event_in_sector = event_in_sector[event_in_sector["sub_industry"] == event_sub_industry]

        event_symbol_options = event_in_sector["symbol"].tolist()
        event_symbol_labels = {row["symbol"]: f"{row['symbol']} - {row['name']}" for _, row in event_in_sector.iterrows()}
        default_symbol_ix = event_symbol_options.index("CRWD") if "CRWD" in event_symbol_options else 0
        event_symbol = st.selectbox(
            "Symbol", options=event_symbol_options, index=default_symbol_ix,
            format_func=lambda s: event_symbol_labels.get(s, s), key="event_symbol",
        ) if event_symbol_options else None

    c3, c4 = st.columns(2)
    event_date_input = c3.date_input(
        "Event date", value=pd.Timestamp("2024-07-19"),
        min_value=pd.Timestamp("2003-01-01"), max_value=pd.Timestamp.today(), key="event_date",
    )
    event_date = event_date_input.strftime("%Y-%m-%d")

    BENCHMARKS = {
        "SPY - S&P 500": "SPY", "QQQ - Nasdaq 100": "QQQ", "DIA - Dow Jones": "DIA", "IWM - Russell 2000": "IWM",
        "XLK - Technology": "XLK", "XLF - Financials": "XLF", "XLE - Energy": "XLE", "XLV - Health Care": "XLV",
        "XLY - Consumer Discretionary": "XLY", "XLP - Consumer Staples": "XLP", "XLI - Industrials": "XLI",
        "XLB - Materials": "XLB", "XLU - Utilities": "XLU", "XLRE - Real Estate": "XLRE", "XLC - Communication Services": "XLC",
    }
    benchmark_label = c4.selectbox("Baseline / benchmark", options=list(BENCHMARKS.keys()), key="event_baseline")
    event_baseline = BENCHMARKS[benchmark_label]

    c5, c6 = st.columns(2)
    event_window = c5.slider("Event window (trading days each side)", 1, 10, DEFAULT_WINDOW)
    event_estimation_days = c6.slider("Estimation window (prior trading days)", 30, 250, DEFAULT_ESTIMATION_DAYS)

    if st.button("Run event study", disabled=not event_symbol):
        try:
            result = event_study(event_symbol, event_date, event_baseline, event_window, event_estimation_days)
        except SystemExit as e:
            st.error(str(e))
            result = None

        if result:
            badge = "✅ statistically reliable (|t| ≥ 2)" if result["reliable"] else "⚠️ not yet reliable (|t| < 2) - treat as exploratory"
            c1, c2, c3 = st.columns(3)
            c1.metric("Cumulative Abnormal Return (CAR)", f"{result['car_pct']:+.2f}%")
            c2.metric("t-stat", result["t_stat"])
            c3.metric("Market-model beta", result["beta"])
            st.caption(f"Fit on {result['n_estimation_days']} prior trading days vs {result['baseline']} "
                       f"(alpha={result['alpha']}) · {badge}")

            window_df = result["event_window"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=window_df["date"], y=window_df["abnormal_ret"],
                                  marker_color=[DOWN if v < 0 else UP for v in window_df["abnormal_ret"]],
                                  name="Abnormal return"))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"{result['symbol']} daily abnormal return vs {result['baseline']} - around {result['event_date']}"),
                               height=460,
                               xaxis_title="Date", yaxis_title="Abnormal return (%, actual - market-model expected)")
            fig.update_xaxes(type="category", tickangle=-45, tickfont=dict(color=INK, size=12))
            fig.update_yaxes(ticksuffix="%", tickfont=dict(color=INK, size=13))
            fig.update_layout(margin=dict(t=60, l=60, r=10, b=100))
            st.plotly_chart(fig, use_container_width=True, theme=None)

            st.dataframe(
                window_df.rename(columns={
                    "date": "Date", "ret_sym": f"{result['symbol']} Return %", "ret_base": f"{result['baseline']} Return %",
                    "expected_ret": "Market-Model Expected %", "abnormal_ret": "Abnormal Return %",
                }),
                use_container_width=True, hide_index=True,
            )
            with st.expander("ℹ️ What is an event study / abnormal return?"):
                st.markdown(
                    "A stock usually moves WITH the market, by an amount set by its **beta** (how sensitive it "
                    "is to the benchmark's moves). The **market model** here is fit on the trading days BEFORE "
                    "the event window, so it captures the stock's normal relationship to the benchmark without "
                    "the event itself skewing the fit.\n\n"
                    "**Abnormal Return (AR)** on a given day = the stock's actual return minus what the market "
                    "model expected from that day's benchmark move alone. **Cumulative Abnormal Return (CAR)** "
                    "sums AR across the event window - the total move that isn't explained by 'the market was "
                    "up/down that day.' The t-stat tests whether that CAR is distinguishable from noise; |t| < 2 "
                    "means treat it as exploratory, not a confirmed finding."
                )

    st.divider()
    st.subheader("Event library")
    st.caption("Everything above looks at ONE event. Logging events of the same type here lets the next two "
               "sections compare across many of them - a cross-event signature, and a check of whether a "
               "live, unresolved case resembles that signature so far.")

    with st.expander("➕ Add an event to the library"):
        el1, el2, el3 = st.columns(3)
        lib_symbol = el1.text_input("Symbol", value="", key="lib_add_symbol").upper()
        lib_event_type = el2.selectbox("Event type", options=EVENT_TYPES, key="lib_add_type")
        lib_event_date = el3.date_input("Event date", value=pd.Timestamp.today(), key="lib_add_date")
        lib_baseline_label = st.selectbox("Baseline / benchmark", options=list(BENCHMARKS.keys()), key="lib_add_baseline")
        lib_notes = st.text_area("Notes (what happened)", key="lib_add_notes")
        if st.button("Add to library", disabled=not lib_symbol):
            add_event(lib_symbol, lib_event_date.strftime("%Y-%m-%d"), lib_event_type, BENCHMARKS[lib_baseline_label], lib_notes)
            st.success(f"Logged {lib_symbol} / {lib_event_date} / {lib_event_type}.")

    with st.expander(f"📒 Current library ({len(load_events())} events)"):
        st.dataframe(load_events(), use_container_width=True, hide_index=True)

    st.markdown("#### Cross-event signature")
    st.caption("Runs the same event study above once per logged event of a type, then averages the day-by-day "
               "abnormal return across all of them - the typical shape of the reaction for that kind of event.")
    sig_types = sorted(load_events()["event_type"].unique())
    sig_event_type = st.selectbox("Event type", options=sig_types, key="sig_event_type")
    if st.button("Build signature"):
        try:
            sig_result = event_sample_study(sig_event_type)
        except SystemExit as e:
            st.error(str(e))
            sig_result = None
        if sig_result:
            s1, s2, s3 = st.columns(3)
            s1.metric("Events in sample", sig_result["n_events"])
            s2.metric("Reliable (|t| ≥ 2)", f"{sig_result['n_reliable']}/{sig_result['n_events']}")
            s3.metric("Average CAR", f"{sig_result['avg_car_pct']:+.2f}%" if sig_result["avg_car_pct"] is not None else "—")
            if sig_result["n_events"] < 4:
                st.warning("Small sample - this signature is a rough average, not a trained model. Add more "
                           "events of this type to firm it up.")

            sig_df = sig_result["signature"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sig_df["relative_day"], y=sig_df["mean_abnormal_ret"],
                                  error_y=dict(array=sig_df["std_abnormal_ret"]),
                                  marker_color=[DOWN if v < 0 else UP for v in sig_df["mean_abnormal_ret"]],
                                  name="Mean abnormal return"))
            fig.update_layout(**PLOTLY_LAYOUT,
                               title=plot_title(f"{sig_event_type} - average abnormal return by day (n={sig_result['n_events']})"),
                               height=420, xaxis_title="Day relative to event (0 = event day)",
                               yaxis_title="Mean abnormal return (%)")
            fig.update_yaxes(ticksuffix="%", tickfont=dict(color=INK, size=13))
            fig.update_layout(margin=dict(t=60, l=60, r=10, b=60))
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.dataframe(sig_df.rename(columns={
                "relative_day": "Day", "mean_abnormal_ret": "Mean Abnormal Return %",
                "std_abnormal_ret": "Std Dev %", "n": "N Events",
            }), use_container_width=True, hide_index=True)
            with st.expander("Per-event scorecard"):
                st.dataframe(sig_result["scorecard"], use_container_width=True, hide_index=True)

    st.markdown("#### Match a live case")
    st.caption("No confirmed event date needed - pick an as-of date (today, or the last close) and see how that "
               "symbol's trajectory so far compares to the historical signature above. This is a resemblance "
               "score, not a prediction: with only a few historical analogs per type, treat it as a lead to "
               "investigate, not a signal to act on by itself.")
    m1, m2, m3 = st.columns(3)
    match_symbol = m1.text_input("Symbol", value="", key="match_symbol").upper()
    match_event_type = m2.selectbox("Suspected event type", options=sig_types, key="match_event_type")
    match_as_of = m3.date_input("As-of date", value=pd.Timestamp.today(), key="match_as_of")
    if st.button("Compare to signature", disabled=not match_symbol):
        try:
            match_result = match_new_case(match_symbol, match_event_type, match_as_of.strftime("%Y-%m-%d"))
        except SystemExit as e:
            st.error(str(e))
            match_result = None
        if match_result:
            mc1, mc2 = st.columns(2)
            mc1.metric("Historical analogs", match_result["n_signature_events"])
            score = match_result["similarity_score"]
            mc2.metric("Similarity score (avg z vs signature)", f"{score:+.2f}" if score is not None else "n/a (sample too small)")
            st.dataframe(match_result["comparison"].rename(columns={
                "relative_day": "Day", "date": "Date", "abnormal_ret": f"{match_symbol} Abnormal Return %",
                "mean_abnormal_ret": "Signature Mean %", "std_abnormal_ret": "Signature Std Dev %",
                "n": "N Events", "z_vs_signature": "Z vs Signature",
            }), use_container_width=True, hide_index=True)
            st.caption("Only days up to and including the as-of date are shown - the days after it haven't "
                       "happened yet, so there's nothing to compare there.")

# ---------------------------------------------------------------------------
with tab_anomaly:
    st.caption("Flags S&P 500 stocks whose trade count today is a statistical outlier against their OWN "
               "recent normal - not against each other. Follows Massive's \"Hunting Anomalies in the Stock "
               "Market\" tutorial: https://massive.com/blog/hunting-anomalies-in-the-stock-market")

    with st.expander("ℹ️ What is this and how does it work?"):
        st.markdown(
            "For each stock, this looks at its last 5 trading days' trade count (number of individual "
            "trades, not shares) and computes **today's z-score** against that rolling mean/std: "
            "`z = (today's trades - 5-day avg) / 5-day std dev`. A stock is flagged when `z` clears the "
            "threshold below (the tutorial's own default is **3 standard deviations**) - i.e. today's "
            "trade activity is far outside what's normal for THIS stock lately, regardless of what the "
            "rest of the market is doing.\n\n"
            "This is a volume/activity signal, not a price signal - price change is shown for context but "
            "never used to decide whether something is flagged. **How this complements After-Market Event "
            "Analysis:** that tool tells you how unusual a PRICE reaction was once you already know (or "
            "suspect) what kind of event happened; this tool tells you something is unusual in trade "
            "activity RIGHT NOW, often before you know why. A ticker flagged here is a reasonable next "
            "candidate to run through the event library's 'Match a live case' above."
        )

    a1, a2 = st.columns(2)
    anomaly_lookback = a1.slider("Lookback window (trading days to pull)", 10, 60, 30, key="anomaly_lookback")
    anomaly_threshold = a2.slider("Z-score threshold", 1.5, 6.0, ANOMALY_DEFAULT_THRESHOLD, 0.5, key="anomaly_threshold")
    anomaly_date_override = st.text_input(
        "Target date override (YYYY-MM-DD, optional - default is the most recent pulled trading day)",
        value="", key="anomaly_date_override",
    )

    if st.button("Scan for anomalies"):
        anomaly_date_clean = anomaly_date_override.strip() or None
        if anomaly_date_clean:
            try:
                pd.Timestamp(anomaly_date_clean)
            except ValueError:
                st.error(f"'{anomaly_date_clean}' isn't a valid date - use YYYY-MM-DD, or clear the field.")
                st.stop()

        with st.spinner("Pulling grouped daily data for the S&P 500 (one call per trading day)..."):
            try:
                anomaly_result = anomaly_scan(anomaly_lookback, anomaly_date_clean, anomaly_threshold)
            except SystemExit as e:
                st.error(str(e))
                anomaly_result = None

        if anomaly_result:
            st.caption(f"Scanned {anomaly_result['n_scanned']} symbols for **{anomaly_result['target_date']}** "
                       f"(threshold z > {anomaly_result['threshold']})")
            flagged = anomaly_result["anomalies"]
            if flagged.empty:
                st.success("No anomalies flagged for this date at this threshold.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=flagged["symbol"], y=flagged["z_score"],
                                      marker_color=[DOWN if v < 0 else UP for v in flagged["price_diff_pct"]],
                                      name="Z-score"))
                fig.update_layout(**PLOTLY_LAYOUT,
                                   title=plot_title(f"Trade-count anomalies - {anomaly_result['target_date']}"),
                                   height=440, xaxis_title="Symbol", yaxis_title="Z-score (trade count vs own 5-day baseline)")
                fig.update_xaxes(type="category", tickangle=-45, tickfont=dict(color=INK, size=12))
                fig.update_yaxes(tickfont=dict(color=INK, size=13))
                fig.update_layout(margin=dict(t=60, l=60, r=10, b=100))
                st.plotly_chart(fig, use_container_width=True, theme=None)

                st.dataframe(flagged.rename(columns={
                    "symbol": "Symbol", "transactions": "Trades Today", "avg_trades": "5-Day Avg Trades",
                    "std_trades": "5-Day Std Dev", "z_score": "Z-Score", "close": "Close",
                    "price_diff_pct": "Price Change %",
                }), use_container_width=True, hide_index=True)
                st.caption("Bar color reflects today's price change (green = up, red = down) - a reminder "
                           "that an activity spike alone doesn't say which direction, if any, price moved.")

# ---------------------------------------------------------------------------
with tab_insider:
    st.caption("Coinbase (COIN) insider Form 3/4 ownership and reported P/S value vs price, plus an optional "
               "single-manager 13F comparison - from the sibling insider-indicator-tool project.")

    with st.expander("ℹ️ What is this and what does the signal mean?"):
        st.markdown(
            "**Form 3** is an insider's initial ownership disclosure; **Form 4** reports every later change, "
            "including open-market/private **P**urchases and **S**ales. This reads Form 4 rows filed in the "
            "chosen quarter, nets known purchase USD minus known sale USD, and compares that net's sign to the "
            "quarter's price change (a ±2% move is treated as flat). When a single manager's SEC CIK and the "
            "security's CUSIP are both given, a 13F quarter-end share comparison for that one manager is added "
            "as a third input.\n\n"
            "This is a **descriptive alignment of reported directions**, not a forecast, a valuation, or an "
            "inferred investor motive - and it covers only one issuer (Coinbase) and, at most, one selected "
            "institutional manager, never market-wide institutional breadth. Missing data becomes 'Insufficient "
            "data' or a partial, two-input reading - it is never filled in with zero or guessed."
        )

    i1, i2, i3 = st.columns(3)
    insider_previous_period = i1.date_input("Previous quarter end", value=pd.Timestamp("2026-03-31"), key="insider_previous_period")
    insider_current_period = i2.date_input("Current quarter end", value=pd.Timestamp("2026-06-30"), key="insider_current_period")
    insider_as_of = i3.date_input("Filing cutoff (as-of)", value=pd.Timestamp.today(), key="insider_as_of")

    with st.expander("Optional: single-manager 13F comparison"):
        im1, im2 = st.columns(2)
        insider_filer_cik = im1.text_input("Manager SEC CIK", value="", key="insider_filer_cik")
        insider_cusip = im2.text_input("Security CUSIP (9 characters)", value="", key="insider_cusip")

    with st.expander("Advanced"):
        ia1, ia2, ia3 = st.columns(3)
        insider_track_purchases = ia1.checkbox("Show individual purchase records", key="insider_track_purchases")
        insider_cluster_days = ia2.slider("Purchase-cluster window (calendar days)", 7, 90, 30, key="insider_cluster_days")
        insider_cluster_buyers = ia3.slider("Purchase-cluster distinct buyers", 2, 10, 3, key="insider_cluster_buyers")
        insider_mode = st.radio(
            "Data source", ["Use cache when fresh (default)", "Force refresh", "Offline (cached data only)"],
            horizontal=True, key="insider_mode",
        )

    if st.button("Run ownership analysis"):
        if insider_previous_period >= insider_current_period:
            st.error("Previous quarter end must be before the current quarter end.")
            st.stop()

        insider_indicator.DATA_CLIENT = insider_indicator.MassiveClient(
            INSIDER_CACHE_DIR,
            offline=insider_mode.startswith("Offline"),
            refresh=insider_mode.startswith("Force"),
            debug=False,
        )
        insider_args = SimpleNamespace(
            previous_period=insider_previous_period.strftime("%Y-%m-%d"),
            current_period=insider_current_period.strftime("%Y-%m-%d"),
            as_of=insider_as_of.strftime("%Y-%m-%d"),
            filer_cik=insider_filer_cik.strip() or None,
            cusip=insider_cusip.strip() or None,
            track_purchases=insider_track_purchases,
            cluster_days=insider_cluster_days,
            cluster_buyers=insider_cluster_buyers,
        )
        with st.spinner("Reading Form 3/4, price and (if configured) 13F from Massive..."):
            try:
                insider_report = insider_indicator.run_pipeline(insider_args)
            except ValueError as e:
                st.error(str(e))
                insider_report = None

        if insider_report:
            result = insider_report["classification"]
            price, form4 = insider_report["price"], insider_report["form4"]
            stale_badge = " · ⚠️ includes stale/cached data" if result.get("uses_stale_data") else ""
            st.subheader(f"{result['signal']}{stale_badge}")
            st.caption(result["rule"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Price change (quarter)", f"{price['change_pct']:+.2f}%" if price.get("status") == "ok" else "Unavailable")
            net = form4.get("net_insider_value") if isinstance(form4, dict) else None
            c2.metric("Net reported P/S value", f"${net:+,.0f}" if net is not None else "Unavailable")
            inst = insider_report["institutional"]
            c3.metric("Selected-manager share change", f"{inst['share_change']:+,.0f}" if inst.get("status") == "ok" else inst.get("status", "not_configured").replace("_", " "))

            if result.get("counterevidence"):
                st.warning("\n".join(f"- {c}" for c in result["counterevidence"]))

            if price.get("status") == "ok" and price.get("chart"):
                chart_df = pd.DataFrame(price["chart"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_df["day"], y=chart_df["close"], mode="lines", line=dict(color=BLUE), name="Close"))
                fig.update_layout(**PLOTLY_LAYOUT, title=plot_title(f"COIN close - {price['start_date']} to {price['end_date']}"),
                                   height=380, xaxis_title="Date", yaxis_title="Close ($)")
                fig.update_xaxes(type="category", tickangle=-45, tickfont=dict(color=INK, size=11))
                fig.update_layout(margin=dict(t=60, l=60, r=10, b=100))
                st.plotly_chart(fig, use_container_width=True, theme=None)

            st.dataframe(pd.DataFrame([
                {"Layer": "Form 3 (initial ownership)", "Status": insider_report["form3"].get("status", "unavailable"),
                 "Rows": insider_report["form3"].get("rows")},
                {"Layer": "Form 4 (this quarter)", "Status": form4.get("status", "unavailable"), "Rows": form4.get("rows_fetched")},
                {"Layer": "13F (selected manager)", "Status": inst.get("status", "not_configured"), "Rows": None},
            ]), use_container_width=True, hide_index=True)

            if insider_track_purchases and isinstance(form4, dict) and form4.get("p_purchase_tracker", {}).get("transactions"):
                with st.expander("Individual purchase records"):
                    st.dataframe(pd.DataFrame(form4["p_purchase_tracker"]["transactions"]), use_container_width=True, hide_index=True)

            with st.expander("Full report (JSON)"):
                st.json(insider_report)

            st.caption(f"Scope: {result['scope']} · {result['interpretation']}")
