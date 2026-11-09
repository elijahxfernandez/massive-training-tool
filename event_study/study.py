"""Event study - was a stock's reaction around a known event date (earnings, an
outage, a guidance cut...) actually unusual, or just the market moving that day?
Follows Massive's own tutorial for the data-pulling half:
https://massive.com/blog/analyzing-the-market-impact-of-the-crowdstrike-outage

That tutorial itself does NOT compute an abnormal return - it pulls daily OHLCV
via client.list_aggs() (reproduced here almost verbatim, see pull_aggs()) and
just eyeballs a % price-change column in a markdown table. The actual event-
study math below is new, not in the tutorial: a market-model regression, same
"OLS via the normal equations + t-stat reliability gate" approach already used
in dark_pool/analysis.py's fit_beta(), generalized from one earlier CrowdStrike
case study to any symbol/date (e.g. GOOGL or MSFT on an earnings date).

Method (standard event-study market model, e.g. MacKinlay 1997):
  1. Estimation window: `estimation_days` trading days BEFORE the event window,
     fit symbol_return = alpha + beta * baseline_return + e (OLS).
  2. Event window: +/- `window` trading days around the event date.
  3. Abnormal Return (AR) on each day = actual return - (alpha + beta * that
     day's baseline return) - what the stock did beyond what the market model
     predicted from the benchmark's own move that day.
  4. Cumulative Abnormal Return (CAR) = sum of AR over the event window.
  5. t-stat = CAR / (estimation-window AR std-dev * sqrt(window length)) -
     same |t| >= 2 reliability bar used elsewhere in this project.

    uv run python -m event_study.study --symbol CRWD --event-date 2024-07-19
    uv run python -m event_study.study --symbol GOOGL --event-date 2026-10-28 --baseline SPY
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from massive import RESTClient

from shared.config import api_key

DEFAULT_BASELINE = "SPY"
DEFAULT_WINDOW = 3            # trading days each side of the event
DEFAULT_ESTIMATION_DAYS = 120  # trading days used to fit the market model


def pull_aggs(symbol: str, from_: str, to_: str) -> pd.DataFrame:
    """Daily OHLCV for one symbol - Massive's documented client.list_aggs() call,
    unchanged from the tutorial, just wrapped to return a DataFrame instead of
    printing the raw aggregate objects."""
    client = RESTClient(api_key=api_key())
    aggs = list(client.list_aggs(symbol.upper(), 1, "day", from_, to_, limit=50_000))
    if not aggs:
        raise SystemExit(f"No aggregate data for {symbol} between {from_} and {to_}.")
    df = pd.DataFrame([{
        "date": datetime.fromtimestamp(a.timestamp / 1000).strftime("%Y-%m-%d"),
        "open": a.open, "high": a.high, "low": a.low, "close": a.close,
        "volume": a.volume, "transactions": a.transactions,
    } for a in aggs])
    return df.sort_values("date").reset_index(drop=True)


def fit_market_model(symbol_returns: pd.Series, baseline_returns: pd.Series) -> dict:
    """Same OLS-via-normal-equations approach as dark_pool/analysis.py's
    fit_beta(), applied here to daily returns instead of weekly z-scores."""
    x, y = baseline_returns.to_numpy(float), symbol_returns.to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ [alpha, beta]
    dof = len(y) - 2
    resid_std = np.sqrt(np.sum(resid**2) / dof) if dof > 0 else float("nan")
    return {"alpha": float(alpha), "beta": float(beta), "resid_std": float(resid_std), "n": len(y)}


def event_study(symbol: str, event_date: str, baseline: str = DEFAULT_BASELINE,
                 window: int = DEFAULT_WINDOW, estimation_days: int = DEFAULT_ESTIMATION_DAYS) -> dict:
    """The full event study for one symbol/event date. Pulls enough history to
    cover the estimation window + event window, fits the market model on the
    estimation window only (never the event window itself - that would bake
    the event into its own baseline), then measures abnormal return in the
    event window against that model."""
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    # Pull generously (calendar days, not trading days) so we have enough real
    # trading days on both sides even with weekends/holidays in between.
    pull_from = (event_dt - timedelta(days=int(estimation_days * 1.6) + window * 3)).strftime("%Y-%m-%d")
    pull_to = (event_dt + timedelta(days=window * 3)).strftime("%Y-%m-%d")

    sym_df = pull_aggs(symbol, pull_from, pull_to)
    base_df = pull_aggs(baseline, pull_from, pull_to)

    df = sym_df[["date", "close"]].merge(base_df[["date", "close"]], on="date", suffixes=("_sym", "_base"))
    df["ret_sym"] = df["close_sym"].pct_change() * 100
    df["ret_base"] = df["close_base"].pct_change() * 100
    df = df.dropna(subset=["ret_sym", "ret_base"]).reset_index(drop=True)

    event_idx = df["date"].searchsorted(event_date)
    if event_idx >= len(df) or abs((datetime.strptime(df["date"].iloc[min(event_idx, len(df) - 1)], "%Y-%m-%d") - event_dt).days) > 7:
        raise SystemExit(f"Event date {event_date} not found in pulled trading data for {symbol}/{baseline}.")

    event_start, event_end = max(0, event_idx - window), min(len(df), event_idx + window + 1)
    est_start = max(0, event_start - estimation_days)
    estimation = df.iloc[est_start:event_start]
    event = df.iloc[event_start:event_end]

    if len(estimation) < 30:
        raise SystemExit(f"Only {len(estimation)} trading days available to fit the market model "
                          f"(need 30+) - widen the pull window or pick an earlier event date.")

    model = fit_market_model(estimation["ret_sym"], estimation["ret_base"])
    event = event.copy()
    event["relative_day"] = range(event_start - event_idx, event_end - event_idx)
    event["expected_ret"] = model["alpha"] + model["beta"] * event["ret_base"]
    event["abnormal_ret"] = event["ret_sym"] - event["expected_ret"]

    car = event["abnormal_ret"].sum()
    se_car = model["resid_std"] * np.sqrt(len(event)) if model["resid_std"] > 0 else float("nan")
    t_stat = car / se_car if se_car and se_car > 0 else float("nan")

    return {
        "symbol": symbol.upper(), "baseline": baseline.upper(), "event_date": event_date,
        "alpha": round(model["alpha"], 4), "beta": round(model["beta"], 3), "n_estimation_days": model["n"],
        "car_pct": round(float(car), 3), "t_stat": round(float(t_stat), 2) if t_stat == t_stat else None,
        "reliable": bool(abs(t_stat) >= 2) if t_stat == t_stat else False,
        "event_window": event[["date", "relative_day", "ret_sym", "ret_base", "expected_ret", "abnormal_ret"]].round(3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Ticker to analyze, e.g. CRWD, GOOGL, MSFT")
    parser.add_argument("--event-date", required=True, help="Event date, YYYY-MM-DD (e.g. an earnings date or outage date)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help=f"Benchmark symbol (default: {DEFAULT_BASELINE})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help=f"Trading days each side of the event (default: {DEFAULT_WINDOW})")
    parser.add_argument("--estimation-days", type=int, default=DEFAULT_ESTIMATION_DAYS, help=f"Trading days to fit the market model on (default: {DEFAULT_ESTIMATION_DAYS})")
    args = parser.parse_args()

    result = event_study(args.symbol, args.event_date, args.baseline, args.window, args.estimation_days)
    badge = "reliable (|t| >= 2)" if result["reliable"] else "NOT reliable yet (|t| < 2) - treat as exploratory"
    print(f"{result['symbol']} vs {result['baseline']} - event window around {result['event_date']}")
    print(f"Market model fit on {result['n_estimation_days']} prior trading days: "
          f"alpha={result['alpha']}, beta={result['beta']}")
    print(f"Cumulative Abnormal Return (CAR): {result['car_pct']:+.2f}%  t-stat={result['t_stat']}  ({badge})")
    print()
    print(result["event_window"].to_string(index=False))


if __name__ == "__main__":
    main()
