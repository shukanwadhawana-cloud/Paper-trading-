"""Automated paper-trade monitor.

Monitors every OPEN paper trade on each GitHub Actions run, applies the
same trade-manager exit rules, persists the result, and reliably notifies
Telegram of trailing locks, skips, and final outcomes.
"""

import math
from datetime import datetime, timezone

import pandas as pd

from trade_manager import resolve_position_over_bars, TRAIL_STEP_R, EXIT_STILL_OPEN
from market_data import get_provider_for_ticker
from execution_layer import get_default_execution_layer
from paper_trading import update_trailing_notification, mark_close_notified, load_unnotified_closed_trades
from main import send_telegram

MONITOR_INTERVAL = "15m"
ENTRY_VALIDITY_R = 0.20
ENTRY_VALIDITY_SENTINEL = -1.0


def _canonical_ticker(ticker: str) -> str:
    """Normalize ticker aliases before market-data calls."""
    normalized = str(ticker).strip().upper().replace("/", "").replace("-", "").replace("_", "")
    aliases = {
        "XAUTUSDT": "XAUUSDT",
        "XAUTUSD": "XAUUSDT",
        "XAGTUSDT": "XAGUSDT",
        "XAGTUSD": "XAGUSDT",
    }
    return aliases.get(normalized, normalized)


def resolve_open_trade(trade_row: dict, provider) -> dict:
    """Replay an open trade from entry onward using fresh market bars."""
    ticker = _canonical_ticker(trade_row["ticker"])
    direction = trade_row["direction"]
    entry = float(trade_row["entry"])
    sl = float(trade_row["sl"])
    tp_raw = trade_row.get("tp")
    tp = float(tp_raw) if tp_raw not in (None, "") else None
    confidence = trade_row["confidence"]
    entry_time = pd.Timestamp(trade_row["entry_time_utc"])

    df = provider.get_bars(ticker, interval=MONITOR_INTERVAL, start=entry_time)
    if df is None or df.empty:
        return {"status": "STILL_OPEN", "locked_level_r": 0.0, "reason": "no bar data returned this run"}

    position, exit_index = resolve_position_over_bars(
        direction=direction,
        entry=entry,
        sl=sl,
        confidence=confidence,
        df=df,
        entry_index=-1,
        opposite_index=None,
        symbol=ticker,
        tp=tp,
    )
    if position is None:
        return {"status": "STILL_OPEN", "locked_level_r": 0.0, "reason": "re-resolution unexpectedly rejected"}
    if position.exit_reason == EXIT_STILL_OPEN:
        return {"status": "STILL_OPEN", "locked_level_r": position.locked_level_r}

    exit_time = df.index[exit_index]
    return {
        "status": "CLOSED",
        "position": position,
        "duration_bars": exit_index + 1,
        "duration_wall_clock": exit_time - entry_time,
    }


def _maybe_notify_trailing_level(trade_row: dict, locked_level_r: float):
    """Notify every newly reached 0.6R level exactly once."""
    trade_id = trade_row["trade_id"]
    try:
        last_notified = float(trade_row.get("last_notified_level_r") or 0.0)
    except (TypeError, ValueError):
        last_notified = 0.0

    locked = float(locked_level_r or 0.0)
    target = math.floor(locked / TRAIL_STEP_R) * TRAIL_STEP_R
    if target <= 0:
        return

    # -1.0 is the entry-validity sentinel, not a real trailing level.
    if last_notified < 0:
        next_level = TRAIL_STEP_R
    else:
        if target <= last_notified:
            return
        next_level = last_notified + TRAIL_STEP_R

    while next_level <= target + 1e-9:
        next_level = round(next_level, 10)
        msg = (
            f"Trailing Profit Locked — {trade_row['asset']}\n"
            f"Trade ID: {trade_id}\n"
            f"Level: +{next_level:g}R"
        )
        if not send_telegram(msg):
            print(f"{trade_id}: trailing notification failed at +{next_level:g}R; will retry next run.")
            return
        update_trailing_notification(trade_id, next_level)
        trade_row["last_notified_level_r"] = str(next_level)
        print(msg)
        next_level += TRAIL_STEP_R


def _outcome_label(realized_r: float, exit_reason: str) -> str:
    if exit_reason == "ENTRY_VALIDITY_SKIPPED":
        return "SKIPPED"
    if realized_r > 1e-9:
        return "PROFIT"
    if realized_r < -1e-9:
        return "LOSS"
    return "BREAKEVEN"


def _format_close_message(trade_id, trade_row, exit_reason, realized_r, realized_pnl, duration_wall_clock, exit_price=None):
    outcome = _outcome_label(realized_r, exit_reason)
    lines = [
        f"TRADE CLOSED — {outcome} — {trade_row['asset']}",
        f"Trade ID: {trade_id}",
        f"Direction: {trade_row['direction']}",
        f"Realized R: {realized_r:+.2f}R",
        f"Exit reason: {exit_reason}",
    ]
    if exit_price is not None:
        lines.append(f"Exit price: {float(exit_price):.5f}")
    lines.extend([f"Realized PnL: {realized_pnl:+.5f}", f"Duration: {duration_wall_clock}"])
    return "\n".join(lines)


def _close_and_notify(trade_row: dict, result: dict, execution_layer):
    trade_id = trade_row["trade_id"]
    position = result["position"]
    realized_r = position.r_multiple()
    realized_pnl = position.exit_price - position.entry if position.direction == "BUY" else position.entry - position.exit_price

    # If the closing bar had already locked a trailing level, notify that level first.
    _maybe_notify_trailing_level(trade_row, position.locked_level_r)

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
        print(f"{trade_id}: already closed; duplicate close notification suppressed.")
        return

    msg = _format_close_message(
        trade_id, trade_row, position.exit_reason, realized_r,
        realized_pnl, result["duration_wall_clock"], position.exit_price,
    )
    if send_telegram(msg):
        mark_close_notified(trade_id)
        print(msg)
    else:
        print(f"{trade_id}: close notification failed; closed trade will be retried next run.")


def _retry_unnotified_closes():
    """Retry only notifications; never re-close an already CLOSED trade."""
    for row in load_unnotified_closed_trades():
        trade_id = row["trade_id"]
        exit_reason = row.get("exit_reason", "")
        realized_r = float(row.get("realized_r") or 0.0)
        realized_pnl = float(row.get("realized_pnl") or 0.0)
        duration = row.get("duration_wall_clock", "")

        if exit_reason == "ENTRY_VALIDITY_SKIPPED":
            msg = (
                f"TRADE SKIPPED — {row['asset']}\n"
                f"Trade ID: {trade_id}\n"
                f"Direction: {row['direction']}\n"
                f"Planned Entry: {float(row['entry']):.5f}\n"
                f"Current/Check Price: {float(row['exit_price']):.5f}\n"
                f"Reason: Entry Validity Window exceeded"
            )
        else:
            msg = _format_close_message(trade_id, row, exit_reason, realized_r, realized_pnl, duration, row.get("exit_price"))

        if send_telegram(msg):
            mark_close_notified(trade_id)
            print(f"{trade_id}: close/skip notification retry succeeded.")
        else:
            print(f"{trade_id}: close/skip notification retry failed; will retry next run.")


def check_entry_validity(trade_row: dict, current_price: float) -> tuple:
    entry = float(trade_row["entry"])
    sl = float(trade_row["sl"])
    risk = abs(entry - sl)
    current_price = float(current_price)
    if trade_row["direction"] == "BUY":
        boundary = entry + ENTRY_VALIDITY_R * risk
        return current_price <= boundary, boundary
    boundary = entry - ENTRY_VALIDITY_R * risk
    return current_price >= boundary, boundary


def _skip_trade_entry_validity(trade_row, current_price, boundary_price, execution_layer):
    trade_id = trade_row["trade_id"]
    entry = float(trade_row["entry"])
    direction = trade_row["direction"]
    closed = execution_layer.close_trade(
        trade_id=trade_id,
        exit_price=current_price,
        exit_reason="ENTRY_VALIDITY_SKIPPED",
        realized_r=0.0,
        realized_pnl=0.0,
        duration_bars=0,
        duration_wall_clock="0:00:00",
        closed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    if not closed:
        return
    boundary_label = "Maximum Valid Entry" if direction == "BUY" else "Minimum Valid Entry"
    msg = (
        f"TRADE SKIPPED — {trade_row['asset']}\n"
        f"Trade ID: {trade_id}\n"
        f"Direction: {direction}\n"
        f"Planned Entry: {entry:.5f}\n"
        f"Current Price: {float(current_price):.5f}\n"
        f"{boundary_label}: {float(boundary_price):.5f}\n"
        f"Reason: Entry Validity Window exceeded"
    )
    if send_telegram(msg):
        mark_close_notified(trade_id)
        print(msg)
    else:
        print(f"{trade_id}: skip notification failed; will retry next run.")


def manual_close_trade(trade_id: str, exit_price: float, trade_row: dict, execution_layer=None):
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
    entry_time = pd.Timestamp(trade_row["entry_time_utc"])
    result = {
        "status": "CLOSED",
        "position": position,
        "duration_bars": None,
        "duration_wall_clock": pd.Timestamp(datetime.now(timezone.utc)) - entry_time,
    }
    _close_and_notify(trade_row, result, execution_layer)


def run_monitor():
    """Monitor all open trades; one broken trade never stops the others."""
    execution_layer = get_default_execution_layer()
    open_trades = execution_layer.load_open_trades()
    if not open_trades:
        print("Paper Trade Monitor: no open trades to check.")

    for row in open_trades:
        trade_id = row["trade_id"]
        try:
            canonical = _canonical_ticker(row["ticker"])
            provider = get_provider_for_ticker(canonical)
            try:
                last_notified = float(row.get("last_notified_level_r") or 0.0)
            except (TypeError, ValueError):
                last_notified = 0.0

            if last_notified == 0.0:
                try:
                    current_price = provider.get_current_price(canonical)
                except Exception as exc:
                    print(f"{trade_id}: current-price check failed: {exc}; retrying next run.")
                    continue
                valid, boundary = check_entry_validity(row, current_price)
                if not valid:
                    _skip_trade_entry_validity(row, current_price, boundary, execution_layer)
                    continue
                update_trailing_notification(trade_id, ENTRY_VALIDITY_SENTINEL)
                row["last_notified_level_r"] = str(ENTRY_VALIDITY_SENTINEL)

            result = resolve_open_trade(row, provider)
        except Exception as exc:
            print(f"{trade_id}: ERROR while monitoring - {exc}")
            continue

        if result["status"] == "STILL_OPEN":
            _maybe_notify_trailing_level(row, result.get("locked_level_r", 0.0))
        else:
            _close_and_notify(row, result, execution_layer)

    _retry_unnotified_closes()


if __name__ == "__main__":
    run_monitor()
