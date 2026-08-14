"""
Paper Trade Monitor - the fully automated paper-trading engine.

Every scheduled run:
1. Loads all OPEN trades through execution_layer.py.
2. Fetches fresh price bars through market_data.py.
3. Re-resolves each position from scratch using the unmodified
   resolve_position_over_bars() in trade_manager.py.
4. Sends exactly one Telegram notification per newly locked trailing
   level and exactly one notification per trade close, with
   deduplication persisted in paper_trades.csv.

KNOWN LIMITATION:
This monitor detects SL and TRAILING_STOP exits from price bars.
It does NOT currently detect opposite-signal exits for already-open
trades because that would require re-running the complete signal
detection + HTF alignment pipeline.

This module deliberately does not modify or import:
- compute_signals
- compute_leg_structure
- BOS/CHoCH detection
- order block detection
- HTF alignment
- session logic
- other strategy-detection logic

Only trade_manager, market_data, execution_layer, paper_trading
notification helpers, and main.send_telegram are used.
"""

import math
from datetime import datetime, timezone

import pandas as pd

from trade_manager import (
    resolve_position_over_bars,
    TRAIL_STEP_R,
    EXIT_STILL_OPEN,
)

from market_data import (
    get_provider_for_ticker,
)

from execution_layer import (
    get_default_execution_layer,
)

from paper_trading import (
    update_trailing_notification,
    mark_close_notified,
    load_unnotified_closed_trades,
)

from main import send_telegram


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MONITOR_INTERVAL = "15m"

# A trade is considered executable only if the current market price
# remains within this fraction of the original risk (R) from entry.
#
# BUY:
#     current price <= entry + 0.20R
#
# SELL:
#     current price >= entry - 0.20R
#
ENTRY_VALIDITY_R = 0.20


# ---------------------------------------------------------------------------
# POSITION RE-RESOLUTION
# ---------------------------------------------------------------------------

def resolve_open_trade(trade_row: dict, provider) -> dict:
    """
    Fetch fresh bars for one open trade and completely re-resolve its
    position from the original entry onward.

    IMPORTANT:
    trade_manager.resolve_position_over_bars() is imported UNMODIFIED.

    entry_index=-1 is intentional.

    The market-data abstraction returns bars strictly after the supplied
    entry timestamp, so there is no entry bar to skip. Therefore -1 causes
    resolve_position_over_bars() to examine every returned bar.
    """

    ticker = trade_row["ticker"]
    direction = trade_row["direction"]

    entry = float(trade_row["entry"])
    sl = float(trade_row["sl"])
    confidence = trade_row["confidence"]

    entry_time = pd.Timestamp(
        trade_row["entry_time_utc"]
    )

    # -----------------------------------------------------------------------
    # Fetch fresh market data
    # -----------------------------------------------------------------------

    df = provider.get_bars(
        ticker,
        interval=MONITOR_INTERVAL,
        start=entry_time,
    )

    if df is None or df.empty:
        return {
            "status": "STILL_OPEN",
            "locked_level_r": 0.0,
            "reason": "no bar data returned this run",
        }

    # -----------------------------------------------------------------------
    # Stateless position replay
    # -----------------------------------------------------------------------

    position, exit_index = resolve_position_over_bars(
        direction=direction,
        entry=entry,
        sl=sl,
        confidence=confidence,
        df=df,
        entry_index=-1,
        opposite_index=None,
    )

    if position is None:
        return {
            "status": "STILL_OPEN",
            "locked_level_r": 0.0,
            "reason": "re-resolution unexpectedly rejected",
        }

    # -----------------------------------------------------------------------
    # Still open
    # -----------------------------------------------------------------------

    if position.exit_reason == EXIT_STILL_OPEN:
        return {
            "status": "STILL_OPEN",
            "locked_level_r": position.locked_level_r,
        }

    # -----------------------------------------------------------------------
    # Closed
    # -----------------------------------------------------------------------

    exit_time = df.index[exit_index]

    # Because entry_index=-1, exit_index represents the number of bars
    # elapsed since the entry.
    duration_bars = exit_index + 1

    duration_wall_clock = exit_time - entry_time

    return {
        "status": "CLOSED",
        "position": position,
        "duration_bars": duration_bars,
        "duration_wall_clock": duration_wall_clock,
    }


# ---------------------------------------------------------------------------
# TRAILING-LEVEL NOTIFICATION
# ---------------------------------------------------------------------------

def _maybe_notify_trailing_level(
    trade_row: dict,
    locked_level_r: float,
):
    """
    Send a Telegram message only when a NEW trailing level has been locked.

    Deduplication is based on last_notified_level_r persisted in
    paper_trades.csv.
    """

    trade_id = trade_row["trade_id"]

    try:
        last_notified = float(
            trade_row.get("last_notified_level_r") or 0.0
        )
    except (TypeError, ValueError):
        last_notified = 0.0

    locked_level_r = float(locked_level_r or 0.0)

    # Nothing new to notify.
    if locked_level_r <= last_notified:
        return

    if locked_level_r <= 0:
        return

    # Normalize to the strategy's configured trailing step.
    new_level = (
        math.floor(locked_level_r / TRAIL_STEP_R)
        * TRAIL_STEP_R
    )

    # The normalized level is not actually newer.
    if new_level <= last_notified:
        return

    msg = (
        f"Trailing profit locked — {trade_row['asset']}\n"
        f"Trade ID: {trade_id}\n"
        f"Locked level: +{new_level:g}R"
    )

    if send_telegram(msg):
        update_trailing_notification(
            trade_id,
            new_level,
        )

        print(msg)

    else:
        print(
            f"{trade_id}: WARNING - trailing-level notification "
            f"failed to send. Will retry next run "
            f"(last_notified_level_r not updated)."
        )


# ---------------------------------------------------------------------------
# CLOSE + NOTIFY
# ---------------------------------------------------------------------------

def _close_and_notify(
    trade_row: dict,
    result: dict,
    execution_layer,
):
    """
    Persist the close through the execution layer and send exactly one
    close notification.

    close_trade() provides the idempotency guard.
    """

    trade_id = trade_row["trade_id"]
    position = result["position"]

    realized_r = position.r_multiple()

    # Price-unit P&L.
    #
    # No position quantity exists in the current paper-trading model,
    # therefore this deliberately does NOT fabricate an account-currency
    # P&L.
    if position.direction == "BUY":
        realized_pnl = (
            position.exit_price - position.entry
        )
    else:
        realized_pnl = (
            position.entry - position.exit_price
        )

    closed = execution_layer.close_trade(
        trade_id=trade_id,
        exit_price=position.exit_price,
        exit_reason=position.exit_reason,
        realized_r=realized_r,
        realized_pnl=realized_pnl,
        duration_bars=result["duration_bars"],
        duration_wall_clock=str(
            result["duration_wall_clock"]
        ),
        closed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    # Idempotency guard.
    if not closed:
        print(
            f"{trade_id}: already closed, "
            f"skipping duplicate close/notification."
        )
        return

    _send_close_message(
        trade_id=trade_id,
        trade_row=trade_row,
        position=position,
        realized_r=realized_r,
        realized_pnl=realized_pnl,
        duration_wall_clock=result["duration_wall_clock"],
    )


def _send_close_message(
    trade_id,
    trade_row,
    position,
    realized_r,
    realized_pnl,
    duration_wall_clock,
):
    """
    Send the trade-close Telegram notification.

    close_notified is updated ONLY after Telegram successfully accepts
    the message.
    """

    msg = (
        f"Trade Closed — {trade_row['asset']}\n"
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
        print(
            f"{trade_id}: WARNING - close notification failed "
            f"to send. Will retry next run "
            f"(trade remains CLOSED; only notification retries)."
        )


# ---------------------------------------------------------------------------
# RETRY FAILED CLOSE NOTIFICATIONS
# ---------------------------------------------------------------------------

def _retry_unnotified_closes():
    """
    Retry Telegram notifications for trades that are already CLOSED
    but whose close notification was not successfully sent.

    This does NOT call close_trade() again and therefore cannot recreate
    or duplicate a trade close.
    """

    rows = load_unnotified_closed_trades()

    for row in rows:

        trade_id = row["trade_id"]

        exit_reason = row["exit_reason"]

        realized_r = float(
            row["realized_r"]
        )

        realized_pnl = float(
            row["realized_pnl"]
        )

        duration_wall_clock = row[
            "duration_wall_clock"
        ]

        msg = (
            f"Trade Closed — {row['asset']}\n"
            f"Trade ID: {trade_id}\n"
            f"{row['direction']}\n"
            f"Exit reason: {exit_reason}\n"
            f"Realized R: {realized_r:.2f}\n"
            f"Realized PnL: {realized_pnl:.5f}\n"
            f"Duration: {duration_wall_clock}"
        )

        if send_telegram(msg):

            mark_close_notified(trade_id)

            print(
                f"{trade_id}: retried close notification succeeded."
            )

        else:

            print(
                f"{trade_id}: retried close notification failed again. "
                f"Will retry next run."
            )


# ---------------------------------------------------------------------------
# FUTURE MANUAL-CLOSE EXTENSION
# ---------------------------------------------------------------------------

def manual_close_trade(
    trade_id: str,
    exit_price: float,
    trade_row: dict,
    execution_layer=None,
):
    """
    Extension point for a future manual-close command.

    Not currently wired to Telegram, UI, webhook, or another trigger.

    Uses the same close/notification pipeline as automatic exits.
    """

    from trade_manager import Position

    if execution_layer is None:
        execution_layer = get_default_execution_layer()

    position = Position(
        direction=trade_row["direction"],
        entry=float(trade_row["entry"]),
        sl=float(trade_row["sl"]),
        confidence=trade_row["confidence"],
    )

    position.manual_close(exit_price)

    entry_time = pd.Timestamp(
        trade_row["entry_time_utc"]
    )

    now = pd.Timestamp(
        datetime.now(timezone.utc)
    )

    result = {
        "status": "CLOSED",
        "position": position,
        "duration_bars": None,
        "duration_wall_clock": now - entry_time,
    }

    _close_and_notify(
        trade_row,
        result,
        execution_layer,
    )


# ---------------------------------------------------------------------------
# ENTRY VALIDITY
# ---------------------------------------------------------------------------

def check_entry_validity(
    trade_row: dict,
    current_price: float,
) -> tuple:
    """
    Determine whether the current market price is still close enough
    to the intended entry to execute the paper trade.

    BUY:
        current_price <= entry + ENTRY_VALIDITY_R * risk

    SELL:
        current_price >= entry - ENTRY_VALIDITY_R * risk

    Returns:

        (is_valid, boundary_price)
    """

    direction = trade_row["direction"]

    entry = float(
        trade_row["entry"]
    )

    sl = float(
        trade_row["sl"]
    )

    risk = abs(
        entry - sl
    )

    current_price = float(
        current_price
    )

    if direction == "BUY":

        boundary_price = (
            entry
            + ENTRY_VALIDITY_R * risk
        )

        return (
            current_price <= boundary_price,
            boundary_price,
        )

    else:

        boundary_price = (
            entry
            - ENTRY_VALIDITY_R * risk
        )

        return (
            current_price >= boundary_price,
            boundary_price,
        )


# ---------------------------------------------------------------------------
# ENTRY VALIDITY SKIP
# ---------------------------------------------------------------------------

def _skip_trade_entry_validity(
    trade_row: dict,
    current_price: float,
    boundary_price: float,
    execution_layer,
):
    """
    Mark a trade as skipped because the market moved outside the allowed
    entry-validity window.

    Uses the existing close_trade() idempotency mechanism rather than
    introducing another persistence mechanism.
    """

    trade_id = trade_row["trade_id"]

    direction = trade_row["direction"]

    entry = float(
        trade_row["entry"]
    )

    if direction == "BUY":
        boundary_label = "Maximum Valid Entry"
    else:
        boundary_label = "Minimum Valid Entry"

    closed = execution_layer.close_trade(
        trade_id=trade_id,
        exit_price=current_price,
        exit_reason="ENTRY_VALIDITY_SKIPPED",
        realized_r=0.0,
        realized_pnl=0.0,
        duration_bars=0,
        duration_wall_clock="0:00:00",
        closed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    if not closed:

        print(
            f"{trade_id}: already closed, "
            f"skipping duplicate skip-notification."
        )

        return

    msg = (
        f"Trade Skipped — {trade_row['asset']}\n"
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

        print(
            f"{trade_id}: WARNING - skip notification failed "
            f"to send. Will retry next run via "
            f"_retry_unnotified_closes()."
        )


# ---------------------------------------------------------------------------
# MAIN MONITOR LOOP
# ---------------------------------------------------------------------------

def run_monitor():
    """
    Main monitor execution.

    One trade failing must never stop monitoring of other trades.
    """

    execution_layer = get_default_execution_layer()

    open_trades = (
        execution_layer.load_open_trades()
    )

    if not open_trades:

        print(
            "Paper Trade Monitor: no open trades to check."
        )

    else:

        for row in open_trades:

            trade_id = row["trade_id"]

            try:

                # -----------------------------------------------------------
                # Provider selection
                # -----------------------------------------------------------

                provider = get_provider_for_ticker(
                    row["ticker"]
                )

                # -----------------------------------------------------------
                # Entry validity
                #
                # IMPORTANT:
                # Normalize the CSV value to float.
                #
                # The old comparison:
                #
                #     row.get(...) == "0"
                #
                # could fail if the CSV/loader returned 0 or 0.0 instead
                # of the string "0".
                # -----------------------------------------------------------

                try:

                    last_notified_level = float(
                        row.get(
                            "last_notified_level_r",
                            0,
                        ) or 0
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    last_notified_level = 0.0

                # A value of exactly 0 means the trade has never passed
                # through the entry-validity gate.
                if last_notified_level == 0.0:

                    try:

                        current_price = (
                            provider.get_current_price(
                                row["ticker"]
                            )
                        )

                    except Exception as exc:

                        print(
                            f"{trade_id}: could not fetch current price "
                            f"for entry-validity check this run "
                            f"({exc}). Will retry next run."
                        )

                        continue

                    is_valid, boundary_price = (
                        check_entry_validity(
                            row,
                            current_price,
                        )
                    )

                    if not is_valid:

                        _skip_trade_entry_validity(
                            row,
                            current_price,
                            boundary_price,
                            execution_layer,
                        )

                        continue

                    # -------------------------------------------------------
                    # Mark the entry-validity check as completed.
                    #
                    # -1.0 is a sentinel:
                    #
                    #   0.0  = never checked
                    #  -1.0  = checked and valid
                    #  >=0   = actual trailing level
                    #
                    # This allows the entry-validity check to happen exactly
                    # once while preserving independent trailing-level
                    # notifications.
                    # -------------------------------------------------------

                    update_trailing_notification(
                        trade_id,
                        -1.0,
                    )

                # -----------------------------------------------------------
                # Stateless position replay
                # -----------------------------------------------------------

                result = resolve_open_trade(
                    row,
                    provider,
                )

            except Exception as exc:

                print(
                    f"{trade_id}: ERROR while monitoring - {exc}"
                )

                continue

            # ---------------------------------------------------------------
            # STILL OPEN
            # ---------------------------------------------------------------

            if result["status"] == "STILL_OPEN":

                _maybe_notify_trailing_level(
                    row,
                    result.get(
                        "locked_level_r",
                        0.0,
                    ),
                )

                continue

            # ---------------------------------------------------------------
            # CLOSED
            # ---------------------------------------------------------------

            _close_and_notify(
                row,
                result,
                execution_layer,
            )

    # -----------------------------------------------------------------------
    # Retry notifications for trades that were already closed but whose
    # Telegram message failed previously.
    # -----------------------------------------------------------------------

    _retry_unnotified_closes()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_monitor()