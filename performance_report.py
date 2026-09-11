#!/usr/bin/env python3
"""Generate cumulative and daily paper-trading performance reports."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone

CSV_PATH = "paper_trades.csv"
OUT_MD = "paper_performance_report.md"
DAILY_CSV = "paper_daily_performance.csv"


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def main():
    cutoff = os.getenv("PERFORMANCE_START_DATE", "2026-07-25")
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("date", "") < cutoff:
                continue
            if r.get("status", "").upper() != "CLOSED":
                continue
            rows.append(r)

    # Only realized trades count. Entry-validity skips are reported separately.
    executed = [r for r in rows if r.get("exit_reason") != "ENTRY_VALIDITY_SKIPPED"]
    skipped = [r for r in rows if r.get("exit_reason") == "ENTRY_VALIDITY_SKIPPED"]

    total_r = sum(f(r, "realized_r") for r in executed)
    total_pnl = sum(f(r, "realized_pnl") for r in executed)
    wins = [r for r in executed if f(r, "realized_r") > 0]
    losses = [r for r in executed if f(r, "realized_r") < 0]
    breakeven = [r for r in executed if f(r, "realized_r") == 0]
    gross_profit = sum(f(r, "realized_pnl") for r in wins)
    gross_loss = -sum(f(r, "realized_pnl") for r in losses)
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    # Equity curve in R, ordered by close time where available.
    executed.sort(key=lambda r: r.get("closed_at_utc") or r.get("date", ""))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in executed:
        equity += f(r, "realized_r")
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    daily = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "r": 0.0, "pnl": 0.0, "skipped": 0})
    for r in rows:
        d = r.get("date", "")
        if r.get("exit_reason") == "ENTRY_VALIDITY_SKIPPED":
            daily[d]["skipped"] += 1
        else:
            daily[d]["trades"] += 1
            rr = f(r, "realized_r")
            daily[d]["r"] += rr
            daily[d]["pnl"] += f(r, "realized_pnl")
            if rr > 0: daily[d]["wins"] += 1
            elif rr < 0: daily[d]["losses"] += 1

    by_asset = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "r": 0.0, "pnl": 0.0})
    for r in executed:
        a = r.get("ticker", r.get("asset", "UNKNOWN"))
        rr = f(r, "realized_r")
        by_asset[a]["trades"] += 1
        by_asset[a]["r"] += rr
        by_asset[a]["pnl"] += f(r, "realized_pnl")
        if rr > 0: by_asset[a]["wins"] += 1
        elif rr < 0: by_asset[a]["losses"] += 1

    def money(x):
        return f"{x:+.5f}"

    lines = [
        "# Paper Trading Performance Report",
        "",
        f"**Period:** {cutoff} → latest recorded trade",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overall",
        f"- Executed trades: **{len(executed)}**",
        f"- Skipped signals: **{len(skipped)}**",
        f"- Wins / losses / breakeven: **{len(wins)} / {len(losses)} / {len(breakeven)}**",
        f"- Win rate: **{(len(wins) / len(executed) * 100) if executed else 0:.2f}%**",
        f"- Total realized R: **{total_r:+.2f}R**",
        f"- Total realized P&L (CSV units): **{money(total_pnl)}**",
        f"- Profit factor: **{profit_factor:.2f}**" if gross_loss else "- Profit factor: **∞** (no realized losses)",
        f"- Average R/trade: **{(total_r / len(executed)) if executed else 0:+.3f}R**",
        f"- Maximum drawdown: **{max_dd:.2f}R**",
        "",
        "## Daily performance",
        "",
        "| Date | Trades | Wins | Losses | Net R | Net P&L | Skipped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(daily):
        x = daily[d]
        lines.append(f"| {d} | {x['trades']} | {x['wins']} | {x['losses']} | {x['r']:+.2f}R | {money(x['pnl'])} | {x['skipped']} |")

    lines += ["", "## Performance by asset", "", "| Asset | Trades | Wins | Losses | Net R | Net P&L |", "|---|---:|---:|---:|---:|---:|"]
    for a in sorted(by_asset):
        x = by_asset[a]
        lines.append(f"| {a} | {x['trades']} | {x['wins']} | {x['losses']} | {x['r']:+.2f}R | {money(x['pnl'])} |")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(DAILY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "trades", "wins", "losses", "net_r", "net_pnl", "skipped"])
        for d in sorted(daily):
            x = daily[d]
            w.writerow([d, x["trades"], x["wins"], x["losses"], f'{x["r"]:.5f}', f'{x["pnl"]:.5f}', x["skipped"]])

    print("=== PAPER TRADING PERFORMANCE ===")
    print(f"Period: {cutoff} -> latest")
    print(f"Executed: {len(executed)} | Skipped: {len(skipped)}")
    print(f"Win rate: {(len(wins)/len(executed)*100) if executed else 0:.2f}%")
    print(f"Total R: {total_r:+.2f}R")
    print(f"Total P&L: {money(total_pnl)}")
    print(f"Profit factor: {profit_factor:.2f}" if gross_loss else "Profit factor: inf")
    print(f"Max drawdown: {max_dd:.2f}R")


if __name__ == "__main__":
    main()
