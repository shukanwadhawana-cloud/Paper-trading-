"""
Backtest Metrics - Phase 3 of the approved backtesting framework.

Purely analytical: takes a list of SimulatedTrade objects (produced by
simulator.py, imported here, never redefined or reimplemented) and
computes performance statistics from their completed results. Contains
no detection, signal generation, SL/TP, state management, Telegram, or
file-writing logic of any kind - this module reads trade results in and
returns numbers out.

Isolation guarantee: no imports from main.py, paper_trading.py, or
data_loader.py. The only cross-module import is SimulatedTrade from
simulator.py, used solely for attribute access - never instantiated,
never mutated here.
"""

from typing import List
import statistics

import pandas as pd

from backtest.simulator import SimulatedTrade

CLOSED_REASONS = {"TP", "SL", "OPPOSITE_SIGNAL"}


def filter_closed_trades(trades: List[SimulatedTrade]) -> List[SimulatedTrade]:
    """
    Requirement: metrics must only be calculated from completed trade
    results. A trade with exit_reason == 'STILL_OPEN' has no final R
    outcome yet (Phase 2 marks it mark-to-market, not resolved) and is
    excluded here so it can never silently distort win rate, profit
    factor, or any other statistic below.
    """
    return [t for t in trades if t.exit_reason in CLOSED_REASONS]


def _sorted_by_exit(trades: List[SimulatedTrade]) -> List[SimulatedTrade]:
    """
    Chronological ordering by exit_time, not entry_time. This is a
    deliberate design choice: simulator.py (Phase 2, approved) allows
    overlapping trades since it measures per-signal expectancy rather
    than a single-position account. A trade's R outcome only becomes
    "known" at its exit, so the equity curve and monthly returns below
    are built in the order results actually resolved - the only ordering
    that produces a coherent running total from an intentionally
    overlapping trade set. Sorting by entry_time instead could show a
    later-opened, earlier-resolved trade's outcome before an
    earlier-opened, later-resolved one - exactly the ordering ambiguity
    this function exists to resolve consistently.
    """
    return sorted(trades, key=lambda t: t.exit_time)


def compute_metrics(trades: List[SimulatedTrade]) -> dict:
    """
    Computes the full metrics set from a list of SimulatedTrade objects.
    Returns a plain dict. No printing, no file writing, no side effects -
    purely a calculation over data already produced by simulator.py.
    """
    total_input_trades = len(trades)
    closed = filter_closed_trades(trades)
    closed_sorted = _sorted_by_exit(closed)
    excluded_still_open = total_input_trades - len(closed)

    total_trades = len(closed)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "trades_still_open_excluded": excluded_still_open,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": None,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": None,
            "expectancy": None,
            "net_r": 0.0,
            "max_drawdown_r": None,
            "equity_curve": pd.Series(dtype=float),
            "monthly_returns": pd.Series(dtype=float),
            "duration_bars_mean": None,
            "duration_bars_median": None,
            "duration_bars_max": None,
            "duration_wallclock_mean": None,
            "duration_wallclock_median": None,
            "duration_wallclock_max": None,
            "avg_mae": None,
            "avg_mfe": None,
        }

    wins = [t for t in closed if t.r_multiple > 0]
    losses = [t for t in closed if t.r_multiple < 0]
    breakeven = [t for t in closed if t.r_multiple == 0]

    win_rate = len(wins) / total_trades

    gross_profit = sum(t.r_multiple for t in wins)
    gross_loss = abs(sum(t.r_multiple for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    net_r = sum(t.r_multiple for t in closed)
    expectancy = net_r / total_trades

    # --- Equity curve & max drawdown (chronological by exit_time) ---
    r_values = [t.r_multiple for t in closed_sorted]
    exit_times = [t.exit_time for t in closed_sorted]

    cumulative = []
    running = 0.0
    for r in r_values:
        running += r
        cumulative.append(running)
    equity_curve = pd.Series(
        cumulative, index=pd.Index(exit_times, name="exit_time"), name="cumulative_r"
    )

    # max_drawdown_r is reported as a positive number: the largest peak-to-
    # trough decline observed in the cumulative R curve. 0 means equity
    # never fell below a prior peak.
    peak = float("-inf")
    max_dd = 0.0
    for val in cumulative:
        if val > peak:
            peak = val
        drawdown = peak - val
        if drawdown > max_dd:
            max_dd = drawdown
    max_drawdown_r = max_dd

    # --- Monthly returns: grouped by the month the trade's result was
    # realized (exit_time), not by entry month ---
    monthly_df = pd.DataFrame({"r": r_values}, index=pd.Index(exit_times, name="exit_time"))
    monthly_returns = monthly_df["r"].groupby(monthly_df.index.to_period("M")).sum()
    monthly_returns.name = "monthly_r"

    # --- Duration statistics ---
    duration_bars_list = [t.duration_bars for t in closed]
    duration_bars_mean = statistics.mean(duration_bars_list)
    duration_bars_median = statistics.median(duration_bars_list)
    duration_bars_max = max(duration_bars_list)

    wallclock_series = pd.Series([t.duration_wall_clock for t in closed])
    duration_wallclock_mean = wallclock_series.mean()
    duration_wallclock_median = wallclock_series.median()
    duration_wallclock_max = wallclock_series.max()

    # --- MAE / MFE ---
    avg_mae = statistics.mean(t.mae for t in closed)
    avg_mfe = statistics.mean(t.mfe for t in closed)

    return {
        "total_trades": total_trades,
        "trades_still_open_excluded": excluded_still_open,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "net_r": net_r,
        "max_drawdown_r": max_drawdown_r,
        "equity_curve": equity_curve,
        "monthly_returns": monthly_returns,
        "duration_bars_mean": duration_bars_mean,
        "duration_bars_median": duration_bars_median,
        "duration_bars_max": duration_bars_max,
        "duration_wallclock_mean": duration_wallclock_mean,
        "duration_wallclock_median": duration_wallclock_median,
        "duration_wallclock_max": duration_wallclock_max,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
    }
