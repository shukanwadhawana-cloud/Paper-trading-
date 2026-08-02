"""
Paper Trade Monitor - the fully automated paper-trading engine.

Every scheduled run:
1. Loads all OPEN trades (via execution_layer.py's abstraction, not
   paper_trading.py directly - see the module docstring in
   execution_layer.py for why).
2. For each, fetches fresh price bars through the market-data
   abstraction layer (market_data.py) - NEVER calls yfinance directly.
3. Re-resolves the position from scratch using trade_manager.py's
   UNMODIFIED resolve_position_over_bars() - a stateless replay of the
   ENTIRE price history since entry, not incremental state. This is what
   makes restart recovery correct with no special-case logic: a run that
   was skipped for hours produces the exact same result as if it had run
   every 15 minutes throughout that gap, because it always recomputes
   from the same fixed starting point (entry) rather than trusting
   whatever partial state a previous run happened to leave behind.
4. Sends exactly one Telegram notification per newly-locked trailing
   level, and exactly one "trade closed" message per trade close - both
   deduplicated against what's already recorded in paper_trades.csv.

KNOWN LIMITATION, disclosed deliberately: this monitor detects SL and
TRAILING_STOP exits (both derivable from price bars alone). It does NOT
yet detect opposite-signal exits for already-open trades - doing so
would require re-running the full signal-detection pipeline
(compute_signals + HTF alignment) here too, which would mean either
duplicating that orchestration glue a third time (it already exists once
in main.py and once in backtest/simulator.py's _align_htf(), a
previously-flagged, still-open duplication concern) or importing from
the backtest package into live-monitoring code, which is architecturally
backwards. Left as an explicit, named future extension rather than
silently worked around.

Does not modify and does not import: compute_signals, compute_leg_structure,
BOS/CHoCH detection, order block detection, HTF alignment, session logic,
or anything else from main.py's detection surface. Imports only
trade_manager (unmodified) and main.send_telegram (a plain notification
utility, not strategy logic).
"""

import math
from datetime import datetime, timezone

import pandas as pd

from trade_manager import resolve_position_over_bars, TRAIL_STEP_R, EXIT_STILL_OPEN
from market_data import get_default_provider, get_provider_for_ticker
from execution_layer import get_default_execution_layer
from paper_trading import update_trailing_notification, mark_close_notified, load_unnotified_closed_trades
from main import send_telegram

MONITOR_INTERVAL = "15m"  # matches the strategy's signal timeframe throughout this project

# Entry Validity Window (new this milestone): a signal is only executable
# if the current market price remains within ENTRY_VALIDITY_R of the
# intended entry. See check_entry_validity() below. Configurable per the
# explicit instruction.
ENTRY_VALIDITY_R = 0.20


def resolve_open_trade(trade_row: dict, provider) -> dict:
    """
    Fetches fresh bars for one open trade's symbol and re-resolves its
    position from scratch via trade_manager.resolve_position_over_bars()
    (imported unmodified). Returns a dict describing the outcome.

    entry_index=-1 convention: get_bars(start=entry_time) returns only
    bars strictly after entry, so there is no "entry bar" to skip within
    the fetched data - passing entry_index=-1 makes the loop inside
    resolve_position_over_bars() start at range(0, n), examining every
    fetched bar. This works with trade_manager.py's function completely
    unmodified; no special-casing was needed there.

    opposite_index is always None here - see the KNOWN LIMITATION in this
    module's docstring.
    """
    ticker = trade_row["ticker"]
    direction = trade_row["direction"]
    entry = float(trade_row["entry"])
    sl = float(trade_row["sl"])
    confidence = trade_row["confidence"]
    entry_time = pd.Timestamp(trade_row["entry_time_utc"])

    df = provider.get_bars(ticker, interval=MONITOR_INTERVAL, start=entry_time)
    if df is None or df.empty:
        return {"status": "STILL_OPEN", "locked_level_r": 0.0, "reason": "no bar data returned this run"}

    position, exit_index = resolve_position_over_bars(
        direction=direction, entry=entry, sl=sl, confidence=confidence,
        df=df, entry_index=-1, opposite_index=None,
    )

    if position is None:
        # Cannot happen for a trade that was already opened (it was HIGH
        # confidence at entry time, and confidence is not re-evaluated
        # retroactively), but handled defensively rather than assumed.
        return {"status": "STILL_OPEN", "locked_level_r": 0.0, "reason": "re-resolution unexpectedly rejected"}

    if position.exit_reason == EXIT_STILL_OPEN:
        return {"status": "STILL_OPEN", "locked_level_r": position.locked_level_r}

    exit_time = df.index[exit_index]
    duration_bars = exit_index + 1  # entry_index=-1, so exit_index IS the bar count since entry
    duration_wall_clock = exit_time - entry_time

    return {
        "status": "CLOSED",
        "position": position,
        "duration_bars": duration_bars,
        "duration_wall_clock": duration_wall_clock,
    }


def _maybe_notify_trailing_level(trade_row: dict, locked_level_r: float):
    trade_id = trade_row["trade_id"]
    last_notified = float(trade_row.get("last_notified_level_r") or 0.0)

    if locked_level_r <= last_notified or locked_level_r <= 0:
        return  # nothing new to notify - dedup gate

    new_level = math.floor(locked_level_r / TRAIL_STEP_R) * TRAIL_STEP_R
    if new_level <= last_notified:
        return

    msg = (
        f"Trailing profit locked â {trade_row['asset']}\n"
        f"Trade ID: {trade_id}\n"
        f"Locked level: +{new_level:g}R"
    )
    if send_telegram(msg):
        update_trailing_notification(trade_id, new_level)
        print(msg)
    else:
        print(f"{trade_id}: WARNING - trailing-level notification failed to send. Will retry next run "
              f"(last_notified_level_r not updated, so the same level is re-attempted).")


def _close_and_notify(trade_row: dict, result: dict, execution_layer):
    trade_id = trade_row["trade_id"]
    position = result["position"]
    realized_r = position.r_multiple()
    realized_pnl = (
        position.exit_price - position.entry if position.direction == "BUY"
        else position.entry - position.exit_price
    )
    # NOTE: realized_pnl is a price-unit P&L (per 1 unit of the
    # underlying), not an account-currency P&L - no position-sizing or
    # account-balance system exists anywhere in this repository yet, so
    # computing a true dollar/USDT P&L would require fabricating a
    # quantity that was never specified. Disclosed here rather than
    # silently assumed.

    closed = execution_layer.close_trade(
        trade_id=trade_id,
        exit_price=position.exit_price,
        exit_reason=position.exit_reason,
        realized_r=realized_r,
        realized_pnl=realized_pnl,
        duration_bars=result["duration_bars"],
        duration_wall_clock=str(result["duration_wall_clock"]),
        closed_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    if not closed:
        # Already CLOSED (idempotent guard in paper_trading.close_trade())
        # - this run's redundant resolution is silently dropped here,
        # which is exactly what prevents a duplicate close/notification.
        print(f"{trade_id}: already closed, skipping duplicate close/notification.")
        return

    _send_close_message(trade_id, trade_row, position, realized_r, realized_pnl, result["duration_wall_clock"])


def _send_close_message(trade_id, trade_row, position, realized_r, realized_pnl, duration_wall_clock):
    msg = (
        f"Trade Closed â {trade_row['asset']}\n"
        f"Trade ID: {trade_id}\n"
        f"{trade_row['direction']}\n"
        f"Exit reason: {position.exit_reason}\n"
        f"Realized R: {realized_r:.2f}\n"
        f"Realized PnL: {realized_pnl:.5f}\n"
        f"Duration: {duration_wall_clock}"
    )
    if send_telegram(msg):
        mark_close_notified(trade_id)
        print(msg)
    else:
        print(f"{trade_id}: WARNING - close notification failed to send. Will retry next run "
              f"(the trade record stays CLOSED and accurate either way - only the notification retries).")


def _retry_unnotified_closes():
    """Handles the case where a trade closed successfully (CSV updated)
    but the Telegram send failed that run. Since load_open_trades() only
    returns OPEN rows, a CLOSED-but-unnotified trade would otherwise never
    be looked at again - this function is what makes that retry possible
    without ever re-calling close_trade() (so no risk of re-triggering
    resolution logic for an already-closed trade)."""
    for row in load_unnotified_closed_trades():
        trade_id = row["trade_id"]
        exit_reason = row["exit_reason"]
        realized_r = float(row["realized_r"])
        realized_pnl = float(row["realized_pnl"])
        duration_wall_clock = row["duration_wall_clock"]

        msg = (
            f"Trade Closed â {row['asset']}\n"
            f"Trade ID: {trade_id}\n"
            f"{row['direction']}\n"
            f"Exit reason: {exit_reason}\n"
            f"Realized R: {realized_r:.2f}\n"
            f"Realized PnL: {realized_pnl:.5f}\n"
            f"Duration: {duration_wall_clock}"
        )
        if send_telegram(msg):
            mark_close_notified(trade_id)
            print(f"{trade_id}: retried close notification succeeded.")
        else:
            print(f"{trade_id}: retried close notification failed again. Will retry next run.")


def manual_close_trade(trade_id: str, exit_price: float, trade_row: dict, execution_layer=None):
    """
    Extension point for a future manual close command (Change 6 from the
    trade_manager milestone: 'prepare extension points for future manual
    close'). Not wired to any trigger today - no UI, webhook, or command
    anywhere in this repository calls this function. Demonstrates exactly
    how a future manual-close trigger would integrate with the existing
    close/notify pipeline without requiring any further architecture
    changes: construct a Position, call its manual_close(), then reuse
    the same _close_and_notify() path used for automatic exits.
    """
    from trade_manager import Position

    if execution_layer is None:
        execution_layer = get_default_execution_layer()

    position = Position(
        direction=trade_row["direction"], entry=float(trade_row["entry"]),
        sl=float(trade_row["sl"]), confidence=trade_row["confidence"],
    )
    position.manual_close(exit_price)

    entry_time = pd.Timestamp(trade_row["entry_time_utc"])
    now = pd.Timestamp(datetime.now(timezone.utc))
    result = {
        "status": "CLOSED", "position": position,
        "duration_bars": None, "duration_wall_clock": now - entry_time,
    }
    _close_and_notify(trade_row, result, execution_layer)


def check_entry_validity(trade_row: dict, current_price: float) -> tuple:
    """
    Entry Validity Window: a signal is only executable if the current
    market price remains within ENTRY_VALIDITY_R of the intended entry.
    Returns (is_valid: bool, boundary_price: float) - boundary_price is
    the maximum valid entry for a BUY, or the minimum valid entry for a
    SELL, included so the skip notification can report it exactly.

    This is evaluated EXACTLY ONCE per trade - see run_monitor(), which
    gates this check on last_notified_level_r == "0" (the untouched
    record_trade()-set default, meaning this trade has never been
    examined by the monitor before). A trade that already passed this
    check is never re-evaluated against it again, even if price later
    moves further away as the trade progresses normally - that's expected
    behavior as a trade runs, not a reason to retroactively skip it.
    """
    direction = trade_row["direction"]
    entry = float(trade_row["entry"])
    sl = float(trade_row["sl"])
    risk = abs(entry - sl)

    if direction == "BUY":
        boundary_price = entry + ENTRY_VALIDITY_R * risk
        return current_price <= boundary_price, boundary_price
    else:
        boundary_price = entry - ENTRY_VALIDITY_R * risk
        return current_price >= boundary_price, boundary_price


def _skip_trade_entry_validity(trade_row: dict, current_price: float, boundary_price: float, execution_layer):
    """
    Marks a trade SKIPPED (never chased, never converted to a market
    execution) via the EXISTING, unmodified close_trade() - reusing its
    idempotency guarantee and its close_notified retry mechanism rather
    than introducing any new dedup logic. exit_reason is a plain string
    ("ENTRY_VALIDITY_SKIPPED") - paper_trading.py's close_trade() places
    no constraint on which strings are valid, so this required zero
    changes to paper_trading.py or trade_manager.py.
    """
    trade_id = trade_row["trade_id"]
    direction = trade_row["direction"]
    entry = float(trade_row["entry"])
    boundary_label = "Maximum Valid Entry" if direction == "BUY" else "Minimum Valid Entry"

    closed = execution_layer.close_trade(
        trade_id=trade_id, exit_price=current_price, exit_reason="ENTRY_VALIDITY_SKIPPED",
        realized_r=0.0, realized_pnl=0.0, duration_bars=0, duration_wall_clock="0:00:00",
        closed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    if not closed:
        print(f"{trade_id}: already closed, skipping duplicate skip-notification.")
        return

    msg = (
        f"Trade Skipped â {trade_row['asset']}\n"
        f"Trade ID: {trade_id}\n"
        f"{direction}\n"
        f"Planned Entry: {entry:.5f}\n"
        f"Current Price: {current_price:.5f}\n"
        f"{boundary_label}: {boundary_price:.5f}\n"
        f"Reason: Entry Validity Window exceeded"
    )
    if send_telegram(msg):
        mark_close_notified(trade_id)
        print(msg)
    else:
        print(f"{trade_id}: WARNING - skip notification failed to send. Will retry next run "
              f"(via _retry_unnotified_closes(), which already handles any close-notified=False row "
              f"regardless of exit_reason - no new retry logic needed for this case).")


def run_monitor():
    execution_layer = get_default_execution_layer()
    open_trades = execution_layer.load_open_trades()

    if not open_trades:
        print("Paper Trade Monitor: no open trades to check.")
    else:
        for row in open_trades:
            trade_id = row["trade_id"]
            try:
                # PRODUCTION FIX: provider selection now happens INSIDE the
                # try block. Previously it was called before the try/except,
                # meaning any failure here (even though get_provider_for_ticker()
                # is pure/deterministic today and unlikely to fail) would have
                # crashed the entire run instead of just skipping this one
                # trade - a structural gap in the same fault-isolation
                # guarantee already relied on for get_current_price()/
                # resolve_open_trade() below. Consistent placement closes
                # that gap regardless of whether it's exercised today.
                provider = get_provider_for_ticker(row["ticker"])

                # Entry Validity Window: evaluated exactly once, gated on
                # last_notified_level_r == "0" (never examined before).
                if row.get("last_notified_level_r", "0") == "0":
                    try:
                        current_price = provider.get_current_price(row["ticker"])
                    except Exception as exc:
                        print(f"{trade_id}: could not fetch current price for entry-validity "
                              f"check this run ({exc}). Will retry next run.")
                        continue
                    is_valid, boundary_price = check_entry_validity(row, current_price)
                    if not is_valid:
                        _skip_trade_entry_validity(row, current_price, boundary_price, execution_layer)
                        continue
                    # Passed validation - mark as checked using the SAME
                    # unmodified update_trailing_notification() function,
                    # with -1 as a sentinel that is always less than any
                    # real locked level (which are always >= 0), so the
                    # first genuine trailing level still correctly fires
                    # its own notification later.
                    # Notify once when paper trade becomes active
                    if send_telegram(
                        f"ð Paper Trade Opened â {row['asset']}\nTrade ID: {trade_id}\nDirection: {row['direction']}\nEntry: {float(row['entry']):.5f}\nSL: {float(row['sl']):.5f}\nTP: {float(row.get('tp',0)):.5f}"
                    ):
                        print(f"{trade_id}: open notification sent.")
                    update_trailing_notification(trade_id, -1.0)

                result = resolve_open_trade(row, provider)
            except Exception as exc:
                # One trade's data-fetch/resolution failure must never
                # prevent other open trades from being checked this run -
                # same fault-isolation principle already applied to
                # run_backtest.py's per-symbol loop.
                print(f"{trade_id}: ERROR while monitoring - {exc}")
                continue

            if result["status"] == "STILL_OPEN":
                _maybe_notify_trailing_level(row, result.get("locked_level_r", 0.0))
                continue

            _close_and_notify(row, result, execution_layer)

    _retry_unnotified_closes()


if __name__ == "__main__":
    run_monitor()