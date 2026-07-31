"""
Backtest Orchestrator - Phase 6 of the approved backtesting framework.

This file is the orchestration layer ONLY. It contains no trading logic
of its own - no BOS/CHoCH detection, no ATR, no HTF bias, no session
filtering, no SL/TP calculation. Every trading decision is made entirely
by the already-approved modules imported below (Phases 1-5); this file's
only job is calling them in the right order and passing their outputs
between each other.

Imports are deliberately restricted to exactly what orchestration requires:
Phases 1-5's public functions. No import from main.py, paper_trading.py,
Telegram, state management, or any exchange integration exists anywhere
in this file.
"""

import os
from typing import List, Dict

from backtest.data_loader import load_symbol_data
from backtest.simulator import generate_signals_for_backtest, simulate_trades
from backtest.metrics import compute_metrics
from backtest.research_export import export_research_dataset
from backtest.report import generate_summary, build_comparison_table, save_report

BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirrors main.py's SYMBOLS mapping (ticker + whether the session filter
# applies). Duplicated here as plain CONFIGURATION DATA, not logic -
# main.py is not in this phase's allowed-imports list. If main.py's
# SYMBOLS table ever changes, this needs manual updating to match; flagged
# explicitly in the Phase 6 self-audit as a drift risk, not hidden.
SYMBOLS = {
    "Gold (XAU/USD)":     {"ticker": "GC=F",    "use_sessions": True},
    "Silver (XAG/USD)":   {"ticker": "SI=F",    "use_sessions": True},
    "Bitcoin (BTC/USD)":  {"ticker": "BTC-USD", "use_sessions": False},
    "Ethereum (ETH/USD)": {"ticker": "ETH-USD", "use_sessions": False},
}

FILTER_TYPES = ["ALL", "HIGH_CONFIDENCE", "LOW_CONFIDENCE", "BOS_ONLY", "CHOCH_ONLY"]


def _apply_filter(signals: List[dict], filter_name: str) -> List[dict]:
    """
    Pure list filtering over already-computed signal dicts (each already
    tagged with 'confidence' and 'structure' by compute_signals(), Phase 2).
    Selects among existing results - computes nothing new about price,
    structure, or trend. This is orchestration-level filtering, not a
    reimplementation of detection logic.
    """
    if filter_name == "ALL":
        return list(signals)
    if filter_name == "HIGH_CONFIDENCE":
        return [s for s in signals if s.get("confidence") == "HIGH"]
    if filter_name == "LOW_CONFIDENCE":
        return [s for s in signals if s.get("confidence") == "LOW"]
    if filter_name == "BOS_ONLY":
        return [s for s in signals if s.get("structure") == "BOS"]
    if filter_name == "CHOCH_ONLY":
        return [s for s in signals if s.get("structure") == "CHOCH"]
    raise ValueError(f"Unknown filter_type: {filter_name}")


def _safe_ticker_name(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")


def run_backtest() -> Dict:
    """
    Orchestrates the full backtest pipeline across all configured symbols
    and filter types, reusing the already-approved Phase 1-5 modules for
    every trading decision. Returns the assembled results and comparison
    table, and writes report files to backtest/reports/.
    """
    all_results = []  # [{"symbol", "filter_type", "metrics"}, ...] - Phase 5's exact input contract
    summary_sections = []
    failed_symbols = []  # H2 fix: [{"symbol", "error"}, ...] - recorded, not silently dropped
    total_signals = 0
    total_trades = 0

    for symbol_label, cfg in SYMBOLS.items():
        try:
            ticker = cfg["ticker"]
            use_sessions = cfg["use_sessions"]

            data = load_symbol_data(ticker)

            # Signals generated EXACTLY ONCE per symbol - every filter below
            # reuses this same list, never regenerating signals separately.
            signals = generate_signals_for_backtest(
                data["15m"], data["1h"], data["4h"], use_sessions=use_sessions
            )
            total_signals += len(signals)

            for filter_name in FILTER_TYPES:
                filtered_signals = _apply_filter(signals, filter_name)
                # H1 fix: `signals` (filtered) controls which trades OPEN;
                # `all_signals=signals` here refers to the FULL unfiltered list
                # generated once above - opposite-signal exits always reference
                # the complete production stream, never the filtered subset.
                trades = simulate_trades(data["15m"], filtered_signals, all_signals=signals)
                total_trades += len(trades)

                metrics = compute_metrics(trades)

                # Phase 4 symbol issue, addressed here at the orchestration layer:
                # SimulatedTrade carries no symbol field (Phase 2 is frozen), and
                # export_research_dataset()'s signature is fixed at
                # (trades, output_path) - its internal "symbol" column is still
                # written as "" (that specific limitation is NOT fixed by this
                # file). What IS solved: each symbol+filter combination is
                # exported to its own uniquely-named file, so the symbol is
                # recoverable from the output path even though the CSV's
                # internal symbol column remains blank. This is a workaround at
                # the orchestration layer, not a true fix to Phase 4 - see the
                # Phase 6 self-audit for the explicit distinction.
                export_path = os.path.join(
                    BACKTEST_DIR, f"research_trades_{_safe_ticker_name(ticker)}_{filter_name}.csv"
                )
                export_research_dataset(trades, output_path=export_path)

                # Phase 5 input-contract, packaged exactly as specified.
                result_entry = {"symbol": symbol_label, "filter_type": filter_name, "metrics": metrics}
                all_results.append(result_entry)

                summary_sections.append(
                    f"--- {symbol_label} | {filter_name} ---\n{generate_summary(metrics)}\n"
                )

        except Exception as exc:
            # H2 fix: one symbol's failure (bad data, network issue, etc.)
            # must never discard other symbols' already-completed results.
            # Recorded explicitly, not silently swallowed, and processing
            # continues to the next symbol.
            failed_symbols.append({"symbol": symbol_label, "error": str(exc)})
            continue

    comparison_table = build_comparison_table(all_results)

    failure_section = ""
    if failed_symbols:
        failure_lines = ["", "===== FAILED SYMBOLS (skipped, other symbols unaffected) ====="]
        for f in failed_symbols:
            failure_lines.append(f"  {f['symbol']}: {f['error']}")
        failure_lines.append("=================================================================")
        failure_section = "\n".join(failure_lines) + "\n"

    combined_summary = "\n".join(summary_sections) + failure_section

    written = save_report(
        summary=combined_summary,
        comparison_table=comparison_table,
        output_dir=os.path.join(BACKTEST_DIR, "reports"),
    )

    print(
        f"Backtest complete: {len(SYMBOLS)} symbols configured, "
        f"{len(SYMBOLS) - len(failed_symbols)} succeeded, {len(failed_symbols)} failed. "
        f"{len(FILTER_TYPES)} filter types = {len(all_results)} result sets. "
        f"Total signals generated: {total_signals}. Total trades simulated: {total_trades}. "
        f"Reports written: {list(written.values())}"
        + (f" | FAILED: {[f['symbol'] for f in failed_symbols]}" if failed_symbols else "")
    )

    return {
        "results": all_results,
        "comparison_table": comparison_table,
        "reports_written": written,
        "failed_symbols": failed_symbols,
    }


if __name__ == "__main__":
    run_backtest()
