# Massive Training Tool

Analyst training tools built on Massive market data, with one combined front end
("Signal Desk", `app.py`) tying them together.

## Setup

```bash
uv sync
uv run python -m pytest -q
```

Copy `.env.example` to `.env`:
```
FINRA_CLIENT_ID=...          # free: register at developer.finra.org, create a Public app
FINRA_CLIENT_SECRET=...
MASSIVE_API_KEY=...          # from massive.com/dashboard - Stocks Developer plan or higher
```

## Structure

```
dark_pool/
  scanner.py        - free scanner: FINRA's weekly off-exchange (ATS) data, no subscription needed
  live.py            - live/delayed dark-pool scanner over Massive's WebSocket trade feed
                       (exchange==4, trf_id present), with a per-symbol % of ADV outlier signal
  analysis.py        - participation-rate z-score + reversal/follow-through test on the FINRA data
  finra/
    client.py        - FINRA OAuth2 + query client (schema-discovering, retries 5xx)
    collect.py        - CLI: --discover, --all, --weeks, --refresh

event_study/
  study.py           - after-market event analysis: market-model abnormal return (CAR) around a
                       known event date (earnings, an outage...) vs a benchmark, with a t-stat
                       reliability gate. Data-pull half follows Massive's own CrowdStrike tutorial;
                       the abnormal-return math on top of it is this project's own.
  event_library.py   - a growing CSV catalog of past events (events.csv), plus cross-event
                       comparison on top of study.py: event_sample_study() averages the day-by-day
                       reaction across every logged event of a type (the "signature"); match_new_case()
                       compares a live, unresolved case against that signature.

anomaly/
  detector.py        - trade-count anomaly detector: flags S&P 500 stocks whose trade count today is
                       a z-score outlier against their own trailing 5-day baseline. Follows Massive's
                       "Hunting Anomalies in the Stock Market" tutorial, adapted to pull through the
                       REST grouped-daily-aggregates endpoint instead of bulk Flat Files.

shared/
  config.py          - credentials, folders

app.py               - Signal Desk: the combined Streamlit front end (Dark Pool, FINRA, Analysis,
                       NOI, After-Market Event Analysis, Anomaly Detector tabs)
```

## Running things

```bash
# The combined dashboard (all tools, one URL)
uv run streamlit run app.py

# Free FINRA weekly scanner, on its own
uv run python -m dark_pool.finra.collect --discover   # confirm the field schema first
uv run python -m dark_pool.finra.collect --all --workers 24
uv run python -m dark_pool.finra.collect --refresh     # safe to run anytime after that
uv run python -m dark_pool.scanner --latest --top 20

# Live/delayed Massive dark-pool scanner
uv run python -m dark_pool.live --ticker NVDA TSLA

# After-market event study, on its own
uv run python -m event_study.study --symbol CRWD --event-date 2024-07-19
uv run python -m event_study.study --symbol GOOGL --event-date 2026-10-28 --baseline SPY

# Event library - log an event, build a cross-event signature, match a live case
uv run python -m event_study.event_library --add --symbol SYM --event-date YYYY-MM-DD --event-type cybersecurity_breach
uv run python -m event_study.event_library --signature --event-type cybersecurity_breach
uv run python -m event_study.event_library --match --symbol SYM --event-type cybersecurity_breach --as-of YYYY-MM-DD

# Anomaly detector, on its own
uv run python -m anomaly.detector --scan
```

## How the FINRA data actually works (confirmed against the real API, not just docs)

- The historic dataset (`otcMarket/weeklySummaryHistoric`) can **only** be queried by
  week (`weekStartDate`) or tier - **not by symbol**. Each pull grabs one full week,
  every symbol and firm (5,000-row pages, often 6,000-7,000+ ATS-symbol rows/week),
  filtered client-side down to `summaryTypeCode == ATS_W_SMBL` (security-level dark
  pool rows).
- The historic dataset's window is rolling and lags present day by about a year;
  the most recent weeks come from FINRA's separate, non-historic `weeklySummary`
  dataset instead (`dark_pool.finra.collect --live`).
- FINRA's backend seems to degrade under heavy sustained parallel load. `client.py`
  retries 5xx errors automatically (backoff); a lower `--workers` count helps too.

## Scheduled updates

A Windows Scheduled Task, **`DarkPoolScannerRefresh`**, runs `--refresh` three times a
day (9:30 AM, 12:30 PM, 4:00 PM ET) so the FINRA dataset stays current. It's a
machine-level task, not part of this repo:
```powershell
Get-ScheduledTask -TaskName DarkPoolScannerRefresh
```
