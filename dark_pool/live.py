"""Live dark pool scanner - follows Massive's own tutorial:
https://massive.com/blog/dark-pool-scanner-with-massive

Needs a paid Massive plan with MASSIVE_API_KEY set in .env. Connects to the
15-min delayed feed by default, which is what the Stocks Developer ($79/mo)
plan is entitled to - Massive's server rejects the real-time endpoint
otherwise ("You don't have access [to] real-time data"). Pass --realtime if
you're on Stocks Advanced ($199/mo) or higher.

Raw signal: every US equity trade reported to a FINRA TRF (exchange == 4,
trf_id present) is an off-exchange/dark-pool print. On top of that raw feed,
this tracks each symbol's running participation rate - today's cumulative
dark-pool volume as a % of that symbol's normal (20-day average) daily volume.
Same participation-rate methodology dark_pool/analysis.py uses on FINRA's
weekly data, computed live instead of weekly. Every print (plus that running
%) is logged to data/dark_pool_live/YYYY-MM-DD.csv; once a few weeks of that
accumulate, analysis.py's z-score + reversal-test approach can be pointed at
it the same way it already is at the FINRA data.

    uv run python -m dark_pool.live
    uv run python -m dark_pool.live --ticker NVDA TSLA
    uv run python -m dark_pool.live --min-notional 50000
"""

import argparse
import csv
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from massive import WebSocketClient
from massive.websocket.models import EquityTrade, Feed, Market

from dark_pool.finra.collect import FINRA_DIR
from shared.config import DATA_DIR, api_key

EASTERN = ZoneInfo("America/New_York")
TRF_NAMES = {201: "FINRA/NYSE TRF", 202: "FINRA/NASDAQ TRF Carteret", 203: "FINRA/NASDAQ TRF Chicago"}
REF_DIR = FINRA_DIR.parent.parent / "reference_data" / "daily"
ADV_WINDOW = 20
LIVE_DIR = DATA_DIR / "dark_pool_live"
COLUMNS = ["time_et", "symbol", "price", "size", "notional", "venue", "trf_id",
           "cum_shares_today", "adv20", "participation_pct_of_adv"]


def fmt_time(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=EASTERN).strftime("%H:%M:%S")


def load_adv() -> dict[str, float]:
    """Each symbol's 20-day average daily volume, from reference_data - the
    denominator for 'is today's dark flow unusual for this stock.' Reference
    files are named BRK-B.csv (filesystem-safe); Massive's trade.symbol uses
    BRK.B (dot), so normalize on the way in."""
    adv = {}
    for path in REF_DIR.glob("*.csv"):
        df = pd.read_csv(path).tail(ADV_WINDOW)
        if len(df) >= 5:
            symbol = path.stem.upper().replace("-", ".")
            adv[symbol] = df["volume"].mean()
    return adv


def show_trade(trade: EquityTrade, cum_shares: int, adv: float | None) -> None:
    notional = trade.price * trade.size
    venue = TRF_NAMES.get(trade.trf_id, f"TRF {trade.trf_id}")
    pct = f"{cum_shares / adv * 100:.2f}% of ADV today" if adv else "ADV unknown"
    print(
        f"{fmt_time(trade.timestamp)}  {trade.symbol:6}  "
        f"${trade.price:.2f} x {trade.size:,} = ${notional:,.0f}  {venue}  [{pct}]",
        flush=True,
    )


class DailyCsvWriter:
    """Appends to data/dark_pool_live/YYYY-MM-DD.csv, switching files when the ET
    date changes - same pattern as the NOI tool's collector."""

    def __init__(self):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        self.date = None
        self.file = None
        self.writer = None

    def write(self, row: dict, date: str):
        if date != self.date:
            self.close()
            path = LIVE_DIR / f"{date}.csv"
            is_new = not path.exists() or path.stat().st_size == 0
            self.file = open(path, "a", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
            if is_new:
                self.writer.writeheader()
            self.date = date
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def available_days() -> list[str]:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in LIVE_DIR.glob("*.csv"))


def load_day(date: str) -> pd.DataFrame:
    path = LIVE_DIR / f"{date}.csv"
    if not path.exists():
        raise SystemExit(f"No live data for {date}. Available: {available_days()}")
    return pd.read_csv(path)


def todays_outliers(date: str) -> pd.DataFrame:
    """Which symbols are unusually heavy in dark-pool flow RIGHT NOW, relative to
    every other symbol trading dark today - a cross-sectional z-score instead of
    analysis.py's trailing-weeks z-score, same underlying idea: don't just look at
    the raw number, judge it against what's normal before calling it an outlier."""
    df = load_day(date)
    if df.empty:
        return df
    latest = df.sort_values("time_et").groupby("symbol").last().reset_index()
    latest = latest.dropna(subset=["participation_pct_of_adv"])
    if latest.empty:
        return latest
    mean, std = latest["participation_pct_of_adv"].mean(), latest["participation_pct_of_adv"].std()
    latest["z_score"] = (latest["participation_pct_of_adv"] - mean) / std if std else 0.0
    latest["outlier"] = latest["z_score"].abs() >= 2
    return latest.sort_values("participation_pct_of_adv", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-notional", type=float, default=100_000, help="Minimum notional value in USD (default: 100,000)")
    parser.add_argument("--ticker", nargs="+", help="One or more tickers to watch (default: all)")
    parser.add_argument("--realtime", action="store_true", help="Use the real-time feed (needs Stocks Advanced or higher)")
    args = parser.parse_args()

    subs = [f"T.{t.upper()}" for t in args.ticker] if args.ticker else ["T.*"]
    adv = load_adv()
    cum_shares: dict[str, int] = {}
    cum_date = None
    out = DailyCsvWriter()

    feed = Feed.RealTime if args.realtime else Feed.Delayed
    # max_reconnects=None: same fix noi_tool/collector.py needed - without it,
    # the SDK gives up and exits after a finite number of dropped connections,
    # which an unattended all-day scheduled run will eventually hit.
    client = WebSocketClient(api_key=api_key(), feed=feed, market=Market.Stocks, subscriptions=subs,
                              max_reconnects=None)

    def handler(msgs):
        # One malformed/unexpected message must not take down the whole feed -
        # an uncaught exception here would propagate out of client.run() and
        # kill the entire process.
        nonlocal cum_date
        for m in msgs:
            try:
                if not isinstance(m, EquityTrade) or m.exchange != 4 or not m.trf_id:
                    continue  # exchange==4 + trf_id present = off-exchange/dark pool print
                if m.price * m.size < args.min_notional:
                    continue

                today = datetime.fromtimestamp(m.timestamp / 1000, tz=EASTERN).strftime("%Y-%m-%d")
                if today != cum_date:
                    cum_shares.clear()
                    cum_date = today

                cum_shares[m.symbol] = cum_shares.get(m.symbol, 0) + m.size
                symbol_adv = adv.get(m.symbol)
                show_trade(m, cum_shares[m.symbol], symbol_adv)

                out.write({
                    "time_et": fmt_time(m.timestamp), "symbol": m.symbol,
                    "price": m.price, "size": m.size, "notional": round(m.price * m.size, 2),
                    "venue": TRF_NAMES.get(m.trf_id, f"TRF {m.trf_id}"), "trf_id": m.trf_id,
                    "cum_shares_today": cum_shares[m.symbol], "adv20": symbol_adv,
                    "participation_pct_of_adv": round(cum_shares[m.symbol] / symbol_adv * 100, 3) if symbol_adv else None,
                }, today)
            except Exception as exc:
                print(f"Skipped one malformed message ({type(exc).__name__}): {exc}", flush=True)

    delay_note = "real-time" if args.realtime else "15-min delayed"
    print(f"Dark pool scanner ({delay_note}, Massive) - {', '.join(subs)} - min notional ${args.min_notional:,.0f}", flush=True)
    # Same crash-restart loop as noi_tool/collector.py: if client.run() still
    # exits despite max_reconnects=None (an SDK-internal exception, a hard
    # network failure...), restart the whole feed instead of exiting. A
    # scheduled task's own execution-time limit remains the outer stop for the day.
    try:
        while True:
            try:
                client.run(handle_msg=handler)
                break
            except KeyboardInterrupt:
                raise
            except Exception:
                print(f"\nFeed crashed, restarting in 15s:\n{traceback.format_exc()}", flush=True)
                time.sleep(15)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()


if __name__ == "__main__":
    main()
