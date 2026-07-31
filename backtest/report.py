"""
Backtest Report - Phase 5 of the approved backtesting framework.

Responsible ONLY for formatting and writing already-computed results.
Performs no signal detection, trade simulation, metrics calculation, data
loading, filtering, paper trading, Telegram messaging, exchange
interaction, or state management. Every value this module displays or
writes was already computed elsewhere (backtest.metrics) before reaching
this file.

Isolation guarantee: no import from main.py or paper_trading.py anywhere
in this file. No production function (send_telegram, record_trade,
load_state, save_state, compute_signals, compute_htf_bias, session_mask)
is imported or referenced. The only cross-module import is SimulatedTrade
from backtest.simulator, used solely for type hints - never instantiated
or mutated here.
"""

import os
from typing import Dict, List, Optional

import pandas as pd

from backtest.simulator import SimulatedTrade  # noqa: F401  (type-hint use only)

BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BACKTEST_DIR, "reports")


def _fmt(value, spec: str = "") -> str:
    """
    Formatting helper only - never performs a calculation. Renders None
    as 'N/A' so generate_summary() can't crash on the legitimate None
    values Phase 3's edge-case handling produces (e.g. win_rate=None when
    there are zero closed trades).
    """
    if value is None:
        return "N/A"
    if spec:
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def generate_summary(metrics: dict) -> str:
    """
    Formats an already-computed metrics dict (from backtest.metrics.compute_metrics)
    into a human-readable summary string. Performs no calculation beyond
    string formatting of values already present in `metrics`.
    """
    lines = [
        "===== BACKTEST SUMMARY =====",
        f"Total Trades:            {_fmt(metrics.get('total_trades'))}",
        f"Trades Excluded (Open):  {_fmt(metrics.get('trades_still_open_excluded'))}",
        f"Wins:                    {_fmt(metrics.get('wins'))}",
        f"Losses:                  {_fmt(metrics.get('losses'))}",
        f"Breakeven:               {_fmt(metrics.get('breakeven'))}",
        f"Win Rate:                {_fmt(metrics.get('win_rate'), '.2%') if metrics.get('win_rate') is not None else 'N/A'}",
        f"Gross Profit (R):        {_fmt(metrics.get('gross_profit'), '.3f')}",
        f"Gross Loss (R):          {_fmt(metrics.get('gross_loss'), '.3f')}",
        f"Profit Factor:           {_fmt(metrics.get('profit_factor'), '.3f') if metrics.get('profit_factor') is not None else 'N/A'}",
        f"Expectancy (avg R/trade):{_fmt(metrics.get('expectancy'), '.3f') if metrics.get('expectancy') is not None else 'N/A'}",
        f"Net R:                   {_fmt(metrics.get('net_r'), '.3f')}",
        f"Max Drawdown (R):        {_fmt(metrics.get('max_drawdown_r'), '.3f') if metrics.get('max_drawdown_r') is not None else 'N/A'}",
        f"Average MAE:             {_fmt(metrics.get('avg_mae'), '.3f') if metrics.get('avg_mae') is not None else 'N/A'}",
        f"Average MFE:             {_fmt(metrics.get('avg_mfe'), '.3f') if metrics.get('avg_mfe') is not None else 'N/A'}",
        f"Mean Duration (bars):    {_fmt(metrics.get('duration_bars_mean'), '.2f') if metrics.get('duration_bars_mean') is not None else 'N/A'}",
        f"Median Duration (bars):  {_fmt(metrics.get('duration_bars_median'))}",
        f"Max Duration (bars):     {_fmt(metrics.get('duration_bars_max'))}",
        "=============================",
    ]
    return "\n".join(lines)


def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Accepts multiple already-computed metric dictionaries and reorganizes
    them into a comparison table. Never recomputes any value.

    Input contract: `results` is a list of dicts, each shaped:
        {"symbol": <str>, "filter_type": <str>, "metrics": <dict from compute_metrics()>}
    This pairing is required because compute_metrics() output (by design,
    per Phase 3's isolation guarantee) carries no symbol/filter identity of
    its own - that context belongs to whatever orchestrated the backtest
    run, not to the metrics calculation itself.
    """
    rows = []
    for entry in results:
        m = entry["metrics"]
        rows.append({
            "Symbol": entry.get("symbol"),
            "Filter Type": entry.get("filter_type"),
            "Total Trades": m.get("total_trades"),
            "Win Rate": m.get("win_rate"),
            "Profit Factor": m.get("profit_factor"),
            "Expectancy": m.get("expectancy"),
            "Net R": m.get("net_r"),
            "Max Drawdown": m.get("max_drawdown_r"),
            "Avg MAE": m.get("avg_mae"),
            "Avg MFE": m.get("avg_mfe"),
        })

    columns = [
        "Symbol", "Filter Type", "Total Trades", "Win Rate", "Profit Factor",
        "Expectancy", "Net R", "Max Drawdown", "Avg MAE", "Avg MFE",
    ]
    return pd.DataFrame(rows, columns=columns)


def equity_curve_dataframe(metrics: dict) -> pd.DataFrame:
    """
    Returns the equity curve already contained inside `metrics` as a
    DataFrame. No recalculation, no smoothing, no interpolation - purely
    reshapes the existing pandas Series (indexed by exit_time) into a
    two-column DataFrame for convenient CSV export.
    """
    series = metrics.get("equity_curve")
    if series is None:
        return pd.DataFrame(columns=["exit_time", "cumulative_r"])
    df = series.reset_index()
    df.columns = ["exit_time", "cumulative_r"]
    return df


def monthly_returns_dataframe(metrics: dict) -> pd.DataFrame:
    """
    Returns the monthly_returns Series already contained inside `metrics`
    as a DataFrame. No recomputation - purely reshapes the existing Series
    (indexed by calendar-month Period) into a two-column DataFrame.
    """
    series = metrics.get("monthly_returns")
    if series is None:
        return pd.DataFrame(columns=["month", "monthly_r"])
    df = series.reset_index()
    df.columns = ["month", "monthly_r"]
    df["month"] = df["month"].astype(str)
    return df


def _safe_path(filename: str) -> str:
    resolved = os.path.abspath(os.path.join(REPORTS_DIR, filename))
    if os.path.commonpath([resolved, BACKTEST_DIR]) != BACKTEST_DIR:
        raise ValueError(
            f"save_report() refuses to write outside the backtest/ directory. Requested: {resolved}"
        )
    return resolved


def save_report(
    summary: Optional[str] = None,
    comparison_table: Optional[pd.DataFrame] = None,
    equity_curve: Optional[pd.DataFrame] = None,
    monthly_returns: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Writes whichever already-built artifacts are provided to
    backtest/reports/ (or `output_dir`, if supplied and validated to still
    resolve inside backtest/). Creates the directory if missing. Only
    writes a file for arguments that are not None - nothing is computed
    or inferred here.

    Returns a dict mapping artifact name -> path written, for whichever
    files were actually produced this call.
    """
    target_dir = output_dir if output_dir is not None else REPORTS_DIR
    resolved_dir = os.path.abspath(target_dir)
    if os.path.commonpath([resolved_dir, BACKTEST_DIR]) != BACKTEST_DIR:
        raise ValueError(
            f"save_report() refuses to write outside the backtest/ directory. Requested: {resolved_dir}"
        )

    os.makedirs(resolved_dir, exist_ok=True)
    written = {}

    if summary is not None:
        path = os.path.join(resolved_dir, "summary.txt")
        with open(path, "w") as f:
            f.write(summary)
        written["summary"] = path

    if comparison_table is not None:
        path = os.path.join(resolved_dir, "comparison.csv")
        comparison_table.to_csv(path, index=False)
        written["comparison_table"] = path

    if equity_curve is not None:
        path = os.path.join(resolved_dir, "equity_curve.csv")
        equity_curve.to_csv(path, index=False)
        written["equity_curve"] = path

    if monthly_returns is not None:
        path = os.path.join(resolved_dir, "monthly_returns.csv")
        monthly_returns.to_csv(path, index=False)
        written["monthly_returns"] = path

    return written
