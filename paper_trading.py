"""
Paper Trading Engine - now a fully automated engine, not just an open-
trade logger.

record_trade() (entry side, called by main.py - UNCHANGED signature and
core dedup behavior) plus a new set of functions supporting the Paper
Trade Monitor (paper_trade_monitor.py): load_open_trades(),
close_trade(), update_trailing_notification(), and the
close-notification-retry helpers below.

Deliberately standalone: main.py's signal-detection logic (structure,
order blocks, HTF, sessions) is untouched by this module, and this
module still has no dependency on it beyond receiving already-computed
signal dicts as plain arguments.

Persistence: paper_trades.csv lives in the repo and gets committed back
by the GitHub Actions workflow after every run (same pattern already
used for last_signals.json), so trade history survives across ephemeral
runner restarts.

NOTE ON THE FILE RENAME: this file previously wrote to "trades.csv". It
now writes to "paper_trades.csv", matching this milestone's explicit
naming. This is a clean rename, not an additive change - any pre-existing
trades.csv data from an earlier deployment will NOT automatically appear
in paper_trades.csv; a one-time manual copy/rename would be needed when
deploying this update to a live repo that already has trade history.

Idempotency: record_trade() derives a deterministic trade_id from the
ticker + signal time + direction, and checks it against existing rows
before writing - unchanged from before. close_trade() is separately
idempotent: closing an already-CLOSED trade is a no-op that returns
False, which is what makes duplicate closes structurally impossible
regardless of how many times the monitor examines the same trade.
"""

import os
import csv
from datetime import datetime, timezone

TRADES_FILE = "paper_trades.csv"

TRADE_FIELDS = [
    "trade_id",
    "date",
    "time",
    "asset",
    "ticker",
    "direction",
    "entry",
    "sl",
    "tp",
    "confidence",
    "structure",
    "session_ok",
    "htf_ok",
    "status",
    "opened_at_utc",
    "entry_time_utc",
    "exit_price",
    "exit_reason",
    "realized_r",
    "realized_pnl",
    "duration_bars",
    "duration_wall_clock",
    "closed_at_utc",
    "last_notified_level_r",
    "close_notified",
]


def make_trade_id(ticker: str, signal_time, direction: str) -> str:
    """Deterministic, human-readable ID. Doubles as the natural dedup key -
    the same signal (same ticker/time/direction) always produces the same ID."""
    ts = signal_time.strftime("%Y%m%dT%H%M%S")
    return f"{ticker}-{ts}-{direction}"


def _ensure_file():
    if not os.path.exists(TRADES_FILE):
        # Same atomic-write safety as _write_all_rows() below, applied to
        # first-time file creation for consistency.
        tmp_path = TRADES_FILE + ".tmp"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            writer.writeheader()
        os.replace(tmp_path, TRADES_FILE)


def load_existing_trade_ids() -> set:
    """Read trade IDs already recorded, so record_trade() never writes a
    duplicate row even if called more than once for the same signal."""
    _ensure_file()
    ids = set()
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["trade_id"])
    return ids


def record_trade(ticker: str, label: str, signal: dict) -> str:
    """Append one new virtual trade row for this signal, unless it's already
    been recorded. Returns the trade_id either way, so callers can always
    reference it (e.g. in the Telegram message) regardless of whether this
    call actually wrote a new row.

    `signal` is expected to be one of the dicts already produced by
    compute_signals() in main.py - no reshaping needed at the call site.
    Signature and dedup behavior unchanged from before this milestone -
    main.py's call site (record_trade(ticker, label, latest)) required no
    changes.
    """
    _ensure_file()
    trade_id = make_trade_id(ticker, signal["time"], signal["type"])

    existing_ids = load_existing_trade_ids()
    if trade_id in existing_ids:
        return trade_id  # already recorded - no-op, keeps this safe to call every run

    now = datetime.now(timezone.utc)
    row = {
        "trade_id": trade_id,
        "date": signal["time"].strftime("%Y-%m-%d"),
        "time": signal["time"].strftime("%H:%M:%S%z"),
        "asset": label,
        "ticker": ticker,
        "direction": signal["type"],
        "entry": f"{signal['entry']:.5f}",
        "sl": f"{signal['sl']:.5f}",
        "tp": f"{signal['tp']:.5f}",
        "confidence": signal["confidence"],
        "structure": signal["structure"],
        "session_ok": signal["session_ok"],
        "htf_ok": signal["htf_ok"],
        "status": "OPEN",
        "opened_at_utc": now.isoformat(),
        "entry_time_utc": signal["time"].isoformat(),
        "exit_price": "",
        "exit_reason": "",
        "realized_r": "",
        "realized_pnl": "",
        "duration_bars": "",
        "duration_wall_clock": "",
        "closed_at_utc": "",
        "last_notified_level_r": "0",
        "close_notified": "",
    }

    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writerow(row)

    return trade_id


def _read_all_rows() -> list:
    _ensure_file()
    with open(TRADES_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _write_all_rows(rows: list):
    # PRODUCTION FIX (same bug class as the earlier C2 fix for
    # last_signals.json, applied here to paper_trades.csv): the previous
    # version opened TRADES_FILE directly in write mode, which truncates
    # the file immediately. If the process were killed mid-write (a
    # GitHub Actions runner timeout, cancellation, or host preemption -
    # all real occurrences in CI, and non-zero-probability across ~672
    # runs over 7 days), paper_trades.csv could be left truncated or
    # empty, silently destroying the entire trade ledger. Writing to a
    # temp file in the same directory and then atomically replacing the
    # real file (os.replace() is atomic on both POSIX and Windows)
    # guarantees the file is always either the old complete version or
    # the new complete version, never something in between. This does
    # NOT change the schema, function signature, or any caller-visible
    # behavior - every existing caller (close_trade(), mark_close_notified(),
    # update_trailing_notification()) is unaffected.
    tmp_path = TRADES_FILE + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, TRADES_FILE)


def load_open_trades() -> list:
    """Returns every row with status == 'OPEN', as plain dicts. Used by
    the Paper Trade Monitor every scheduled run to know what to check.
    Never assumes there is only one - returns however many exist,
    across however many symbols."""
    return [row for row in _read_all_rows() if row["status"] == "OPEN"]


def close_trade(trade_id: str, exit_price: float, exit_reason: str,
                 realized_r: float, realized_pnl: float,
                 duration_bars: int, duration_wall_clock: str,
                 closed_at_utc: str) -> bool:
    """
    Updates one OPEN row to CLOSED with its final outcome. Idempotent by
    design: if trade_id is already CLOSED, this is a no-op that returns
    False, rather than re-writing or re-triggering anything - this is the
    mechanism that makes duplicate closes structurally impossible even if
    the monitor re-examines the same trade across multiple runs (e.g.
    after a restart-recovery replay).

    close_notified defaults to "" (not yet notified) whenever a trade is
    freshly closed here - see mark_close_notified() / 
    load_unnotified_closed_trades(), which let the monitor retry a
    Telegram send that failed without ever re-running close_trade() (and
    therefore without any risk of re-detecting/re-closing the trade).
    """
    rows = _read_all_rows()
    updated = False
    for row in rows:
        if row["trade_id"] == trade_id:
            if row["status"] == "CLOSED":
                return False  # already closed - idempotent no-op
            row["status"] = "CLOSED"
            row["exit_price"] = f"{exit_price:.5f}"
            row["exit_reason"] = exit_reason
            row["realized_r"] = f"{realized_r:.5f}"
            row["realized_pnl"] = f"{realized_pnl:.5f}"
            row["duration_bars"] = str(duration_bars)
            row["duration_wall_clock"] = duration_wall_clock
            row["closed_at_utc"] = closed_at_utc
            row["close_notified"] = ""
            updated = True
            break
    if updated:
        _write_all_rows(rows)
    return updated


def mark_close_notified(trade_id: str):
    """Records that the 'trade closed' Telegram message was successfully
    sent for this trade. Only called after a confirmed successful send -
    see paper_trade_monitor.py."""
    rows = _read_all_rows()
    for row in rows:
        if row["trade_id"] == trade_id:
            row["close_notified"] = "TRUE"
            break
    _write_all_rows(rows)


def load_unnotified_closed_trades() -> list:
    """Rows that are CLOSED but whose close notification has not yet been
    confirmed sent (e.g. a prior run's Telegram send failed). The monitor
    retries these each run without ever re-resolving or re-closing them -
    close_trade() is never called again for a row already CLOSED."""
    return [
        row for row in _read_all_rows()
        if row["status"] == "CLOSED" and row.get("close_notified") != "TRUE"
    ]


def update_trailing_notification(trade_id: str, locked_level_r: float):
    """Records that a Telegram notification has been sent for this locked
    trailing level, so a later run never re-sends the same level. Only
    called after a confirmed successful send - see paper_trade_monitor.py."""
    rows = _read_all_rows()
    for row in rows:
        if row["trade_id"] == trade_id:
            row["last_notified_level_r"] = str(locked_level_r)
            break
    _write_all_rows(rows)
