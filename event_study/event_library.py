"""Event library: a small, growing catalog of past events (symbol, date, type) that
turns the single-case event_study() in study.py into a cross-event comparison tool.

study.py answers "was THIS move unusual?" for one symbol/date. This module answers
two further questions once you've logged a few events of the same type:

  1. event_sample_study(event_type) - across every past event tagged with this
     type, what did the reaction typically look like day-by-day (the "signature"),
     and how often was it actually reliable (|t| >= 2)?
  2. match_new_case(...) - for a symbol you suspect is going through a similar
     event RIGHT NOW (no confirmed/past event date needed - just an as-of date,
     e.g. today), how does its abnormal-return trajectory so far compare to that
     signature?

Both reuse study.py's event_study() unchanged - no new statistics are invented
here, this is purely aggregation and comparison across multiple single-event runs.

Caveat that should always travel with the output: with a handful of historical
analogs per event type, the "signature" is a small-sample average, not a trained
model. Every result below carries its sample size (n) so this never gets read as
more confident than it actually is.

    uv run python -m event_study.event_library --add --symbol CRWD --event-date 2024-07-19 --event-type cybersecurity_breach
    uv run python -m event_study.event_library --signature --event-type cybersecurity_breach
    uv run python -m event_study.event_library --match --symbol SYM --event-type cybersecurity_breach --as-of 2026-11-07
"""

import argparse
from pathlib import Path

import pandas as pd

from event_study.study import DEFAULT_BASELINE, DEFAULT_ESTIMATION_DAYS, DEFAULT_WINDOW, event_study

EVENTS_PATH = Path(__file__).resolve().parent / "events.csv"

# Controlled vocabulary - not enforced (free text still works), just what's
# offered in the dashboard dropdown so tags stay consistent as the library grows.
EVENT_TYPES = [
    "cybersecurity_breach", "earnings_beat", "earnings_miss", "guidance_cut",
    "guidance_raise", "regulatory_action", "exec_departure", "fda_decision",
    "product_recall", "merger_announcement", "outage",
]

COLUMNS = ["symbol", "event_date", "event_type", "baseline", "notes"]

# Seeded with the cases already validated earlier against this tool, so the
# library isn't empty on first use.
SEED_EVENTS = [
    {"symbol": "CRWD", "event_date": "2024-07-19", "event_type": "outage", "baseline": "SPY",
     "notes": "Falcon sensor update bricked Windows machines worldwide; CAR confirmed against the tutorial's own published numbers."},
    {"symbol": "GOOGL", "event_date": "2026-10-28", "event_type": "earnings_miss", "baseline": "SPY",
     "notes": "Q3 2026 earnings reaction (reports after close 10-28, reaction shows 10-29); CAR -10.17%, t=-2.22, reliable."},
    {"symbol": "MSFT", "event_date": "2026-10-28", "event_type": "earnings_beat", "baseline": "SPY",
     "notes": "Same earnings date as GOOGL above; CAR -0.95%, t=-0.17, NOT reliable - noise, not a real reaction."},
]


def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        df = pd.DataFrame(SEED_EVENTS, columns=COLUMNS)
        df.to_csv(EVENTS_PATH, index=False)
        return df
    return pd.read_csv(EVENTS_PATH, dtype=str)


def add_event(symbol: str, event_date: str, event_type: str, baseline: str = DEFAULT_BASELINE, notes: str = "") -> pd.DataFrame:
    df = load_events()
    row = {"symbol": symbol.upper(), "event_date": event_date, "event_type": event_type,
           "baseline": baseline.upper(), "notes": notes}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.drop_duplicates(["symbol", "event_date", "event_type"], keep="last")
    df.to_csv(EVENTS_PATH, index=False)
    return df


def event_sample_study(event_type: str, window: int = DEFAULT_WINDOW, estimation_days: int = DEFAULT_ESTIMATION_DAYS) -> dict:
    """Run event_study() over every logged event of this type, then align the
    results on relative_day (day -window..+window relative to EACH event's own
    date) and average. The averaged abnormal-return curve IS the signature."""
    events = load_events()
    rows = events[events["event_type"] == event_type]
    if rows.empty:
        raise SystemExit(f"No events logged with event_type={event_type!r}. Add one with --add first.")

    per_event, windows = [], []
    for _, row in rows.iterrows():
        try:
            result = event_study(row["symbol"], row["event_date"], row["baseline"], window, estimation_days)
        except SystemExit as e:
            per_event.append({"symbol": row["symbol"], "event_date": row["event_date"], "error": str(e)})
            continue
        per_event.append({
            "symbol": result["symbol"], "event_date": result["event_date"],
            "car_pct": result["car_pct"], "t_stat": result["t_stat"], "reliable": result["reliable"],
        })
        w = result["event_window"][["relative_day", "abnormal_ret"]].copy()
        w["symbol"] = result["symbol"]
        windows.append(w)

    scorecard = pd.DataFrame(per_event)
    n_ok = int((~scorecard.get("error", pd.Series(dtype=object)).notna()).sum()) if "error" in scorecard else len(scorecard)
    reliable_ct = int(scorecard["reliable"].sum()) if "reliable" in scorecard and n_ok else 0

    if not windows:
        raise SystemExit(f"Could not pull data for any logged {event_type!r} event (check dates/symbols).")
    all_windows = pd.concat(windows, ignore_index=True)
    signature = (
        all_windows.groupby("relative_day")["abnormal_ret"]
        .agg(mean_abnormal_ret="mean", std_abnormal_ret="std", n="count")
        .reset_index()
        .sort_values("relative_day")
    )
    signature["std_abnormal_ret"] = signature["std_abnormal_ret"].fillna(0.0)

    return {
        "event_type": event_type, "n_events": n_ok, "n_reliable": reliable_ct,
        "avg_car_pct": round(float(scorecard["car_pct"].mean()), 3) if n_ok else None,
        "scorecard": scorecard, "signature": signature.round(3),
    }


def match_new_case(symbol: str, event_type: str, as_of: str, baseline: str = DEFAULT_BASELINE,
                    window: int = DEFAULT_WINDOW, estimation_days: int = DEFAULT_ESTIMATION_DAYS) -> dict:
    """How does SYMBOL's abnormal-return trajectory through as_of compare to the
    historical signature for event_type? as_of does NOT need to be a confirmed
    event date - pass today (or the last closed trading day) for a live, unresolved
    case. Only relative days <= 0 exist yet (the "future" half of the window simply
    hasn't happened), which is exactly the half that matters for a live comparison."""
    signature = event_sample_study(event_type, window, estimation_days)["signature"]

    new_case = event_study(symbol, as_of, baseline, window, estimation_days)
    new_window = new_case["event_window"][["relative_day", "date", "abnormal_ret"]]
    new_window = new_window[new_window["relative_day"] <= 0]  # the future half hasn't happened yet

    merged = new_window.merge(signature, on="relative_day", how="left")
    merged["z_vs_signature"] = (merged["abnormal_ret"] - merged["mean_abnormal_ret"]) / merged["std_abnormal_ret"].replace(0, float("nan"))

    valid_z = merged["z_vs_signature"].dropna()
    similarity_score = round(float(valid_z.mean()), 2) if not valid_z.empty else None

    return {
        "symbol": new_case["symbol"], "event_type": event_type, "as_of": as_of,
        "n_signature_events": int(signature["n"].max()) if not signature.empty else 0,
        "similarity_score": similarity_score,
        "comparison": merged.round(3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", action="store_true", help="Log a new event into the library")
    parser.add_argument("--signature", action="store_true", help="Build the cross-event signature for --event-type")
    parser.add_argument("--match", action="store_true", help="Compare --symbol's current trajectory to the --event-type signature")
    parser.add_argument("--symbol")
    parser.add_argument("--event-date")
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--as-of")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.add:
        add_event(args.symbol, args.event_date, args.event_type, args.baseline, args.notes)
        print(f"Logged {args.symbol} / {args.event_date} / {args.event_type} -> {EVENTS_PATH}")
    elif args.signature:
        result = event_sample_study(args.event_type)
        print(f"Signature for {args.event_type!r}: n={result['n_events']} events, "
              f"{result['n_reliable']} reliable, avg CAR={result['avg_car_pct']}%")
        print(result["signature"].to_string(index=False))
    elif args.match:
        result = match_new_case(args.symbol, args.event_type, args.as_of, args.baseline)
        print(f"{result['symbol']} vs {result['event_type']} signature (n={result['n_signature_events']} historical events)")
        print(f"Similarity score (avg z vs signature): {result['similarity_score']}")
        print(result["comparison"].to_string(index=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
