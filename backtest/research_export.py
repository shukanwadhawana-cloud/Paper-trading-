"""
Backtest Research Export - Phase 4 of the approved backtesting framework.

Responsible ONLY for serializing already-completed SimulatedTrade objects
into a flat CSV research dataset. Performs no trading logic, no signal
detection, no simulation, no metrics calculation, no state management, no
Telegram operations, and no exchange interaction of any kind.

Isolation guarantee: the only cross-module import is SimulatedTrade from
backtest.simulator, used solely for attribute access. No import from
main.py, paper_trading.py, or data_loader.py exists anywhere in this file.
"""

import os
from typing import List, Optional

import pandas as pd

from backtest.simulator import SimulatedTrade

BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(BACKTEST_DIR, "research_trades.csv")

COLUMNS = [
    "symbol",
    "timestamp",
    "direction",
    "structure",
    "confidence",
    "filter_type",
    "window_id",
    "entry",
    "sl",
    "tp",
    "exit_price",
    "exit_reason",
    "r_result",
    "mae",
    "mfe",
    "duration_bars",
    "duration_wall_clock",
]


def export_research_dataset(trades: List[SimulatedTrade], output_path: Optional[str] = None) -> None:
    """
    Serializes `trades` into a CSV research dataset at `output_path`
    (defaults to backtest/research_trades.csv). Overwrites the file on
    every call - no append, no merge with prior runs.

    Every value written is read directly off an existing SimulatedTrade
    attribute. No new trading decision, filter, or calculation is
    performed here - this function only reshapes data that already exists.

    KNOWN LIMITATION (Phase 4, intentionally not resolved without approval):
    SimulatedTrade (Phase 2, frozen) has no `symbol` attribute -
    simulate_trades() is called once per symbol and the symbol itself is
    tracked by the caller, not the trade object. This function cannot
    populate a real value for the `symbol` column from `trades` alone; it
    is written as an empty string for every row. See the Phase 4
    self-audit for the full explanation and options.

    `timestamp` is populated from `entry_time` (the moment the underlying
    signal occurred) - consistent with how the live bot already references
    a signal by its entry time in Telegram messages. This is a naming
    interpretation, not a data gap: entry_time is real, complete data.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    resolved = os.path.abspath(output_path)
    if os.path.commonpath([resolved, BACKTEST_DIR]) != BACKTEST_DIR:
        raise ValueError(
            f"export_research_dataset() refuses to write outside the backtest/ directory. "
            f"Requested: {resolved}"
        )

    rows = []
    for t in trades:
        rows.append({
            "symbol": "",
            "timestamp": t.entry_time,
            "direction": t.direction,
            "structure": t.structure,
            "confidence": t.confidence,
            "filter_type": "ALL",
            "window_id": "full_history",
            "entry": t.entry,
            "sl": t.sl,
            "tp": t.tp,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "r_result": t.r_multiple,
            "mae": t.mae,
            "mfe": t.mfe,
            "duration_bars": t.duration_bars,
            "duration_wall_clock": t.duration_wall_clock,
        })

    df = pd.DataFrame(rows, columns=COLUMNS)

    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    df.to_csv(resolved, index=False)
