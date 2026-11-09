"""Anomaly detector - flags stocks whose trade count today is statistically far
above its own recent normal, following Massive's own tutorial:
https://massive.com/blog/hunting-anomalies-in-the-stock-market

The tutorial builds this from bulk Flat Files (the whole market, every day) and a
local lookup table. This project already has an S&P 500 symbol universe and daily
history pulled elsewhere (market_reference.py / event_study), so instead of
re-downloading flat files this pulls the same per-day, whole-market data through
Massive's REST "grouped daily aggregates" endpoint (one call per day, returns every
US stock's OHLC/transactions for that date) and filters to this project's S&P 500
list - same underlying fields (ticker, close, transactions), far less to download.

Method (unchanged from the tutorial):
  1. For each symbol, shift the trade-count (transactions) series by one day so
     today is never part of its own baseline.
  2. Rolling 5-day mean/std of that shifted series = "what's normal for this stock
     lately."
  3. z = (today's transactions - rolling mean) / rolling std.
  4. Flag z > threshold (tutorial default: 3, i.e. "3 standard deviations").

This complements After-Market Event Analysis rather than replacing it: this tool
answers "is something unusual happening in trade activity RIGHT NOW" (today, no
known cause yet); event_study/event_library answers "once you know (or suspect)
what kind of event this is, how unusual was/is the PRICE reaction, and does it
match how similar past events behaved."

    uv run python -m anomaly.detector --scan
    uv run python -m anomaly.detector --scan --date 2026-11-06 --threshold 3
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from massive import RESTClient

from shared.config import api_key

ROOT = Path(__file__).resolve().parent.parent
CONSTITUENTS_PATH = ROOT / "reference_data" / "sp500_constituents.csv"

DEFAULT_LOOKBACK_DAYS = 30   # trading days of history to pull (need 5+ for the rolling baseline)
DEFAULT_THRESHOLD = 3.0      # tutorial's "e.g. 3 standard deviations"
ROLLING_WINDOW = 5           # tutorial's rolling baseline window


def _universe() -> set[str]:
    if not CONSTITUENTS_PATH.exists():
        raise SystemExit(f"No S&P 500 reference data found at {CONSTITUENTS_PATH}.")
    return set(pd.read_csv(CONSTITUENTS_PATH)["symbol"].str.replace("-", ".", regex=False))


def pull_grouped_history(lookback_days: int = DEFAULT_LOOKBACK_DAYS, end_date: str | None = None) -> pd.DataFrame:
    """One REST call per calendar day (skipping days the market was closed),
    filtered down to the S&P 500 universe. Returns ticker, date, transactions,
    close for every (symbol, trading day) pulled."""
    universe = _universe()
    client = RESTClient(api_key=api_key())
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()

    rows, trading_days_found = [], 0
    calendar_days_back = -1
    while trading_days_found < lookback_days and calendar_days_back < lookback_days * 2.2:
        calendar_days_back += 1
        # back=0 tries `end` itself first - matters when a caller passes an explicit
        # target date (it must be included), harmless when end=today and today's
        # bar isn't closed yet (that attempt just comes back empty and gets skipped).
        day = (end - timedelta(days=calendar_days_back)).strftime("%Y-%m-%d")
        try:
            aggs = client.get_grouped_daily_aggs(day)
        except Exception:
            continue
        if not aggs:
            continue  # weekend/holiday
        trading_days_found += 1
        for a in aggs:
            if a.ticker in universe:
                rows.append({"symbol": a.ticker, "date": day, "transactions": a.transactions, "close": a.close})

    if not rows:
        raise SystemExit("No grouped daily data pulled - check MASSIVE_API_KEY / date range.")
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def build_lookup_table(history: pd.DataFrame) -> pd.DataFrame:
    """Same shape as the tutorial's build-lookup-table.py: per (symbol, date),
    the rolling 5-day avg/std of PRIOR days' trade count, plus today's own
    price_diff for context (not used as a filter, same as the tutorial)."""
    df = history.copy()
    df["price_diff_pct"] = df.groupby("symbol")["close"].pct_change() * 100
    g = df.groupby("symbol")["transactions"]
    trades_shifted = g.shift(1)
    df["avg_trades"] = trades_shifted.groupby(df["symbol"]).transform(lambda s: s.rolling(ROLLING_WINDOW).mean())
    df["std_trades"] = trades_shifted.groupby(df["symbol"]).transform(lambda s: s.rolling(ROLLING_WINDOW).std())
    return df


def detect_anomalies(lookup: pd.DataFrame, target_date: str, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """z = (transactions - avg_trades) / std_trades; flag z > threshold. Sorted by
    trade count descending, same as the tutorial's query-lookup-table.py."""
    day = lookup[lookup["date"] == target_date].copy()
    day = day[day["std_trades"].notna() & (day["std_trades"] > 0)]
    day["z_score"] = (day["transactions"] - day["avg_trades"]) / day["std_trades"]
    flagged = day[day["z_score"] > threshold].sort_values("transactions", ascending=False)
    return flagged[["symbol", "transactions", "avg_trades", "std_trades", "z_score", "close", "price_diff_pct"]].round(2)


def scan(lookback_days: int = DEFAULT_LOOKBACK_DAYS, target_date: str | None = None,
          threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Convenience: pull, build the lookup table, detect for the latest (or given)
    date in the pulled range."""
    history = pull_grouped_history(lookback_days, end_date=target_date)
    lookup = build_lookup_table(history)
    resolved_date = target_date or lookup["date"].max()
    flagged = detect_anomalies(lookup, resolved_date, threshold)
    return {"target_date": resolved_date, "threshold": threshold, "n_scanned": lookup["symbol"].nunique(),
            "anomalies": flagged, "available_dates": sorted(lookup["date"].unique())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: most recent pulled trading day)")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if not args.scan:
        parser.print_help()
        return

    result = scan(args.lookback_days, args.date, args.threshold)
    print(f"Scanned {result['n_scanned']} S&P 500 symbols for {result['target_date']} "
          f"(z > {result['threshold']} flagged as anomalous trade count)")
    if result["anomalies"].empty:
        print("No anomalies flagged.")
    else:
        print(result["anomalies"].to_string(index=False))


if __name__ == "__main__":
    main()
