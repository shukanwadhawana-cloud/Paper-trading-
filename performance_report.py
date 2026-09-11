#!/usr/bin/env python3
"""Generate a full paper-trading performance and strategy-audit report."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

CSV_PATH = "paper_trades.csv"
OUT_MD = "paper_performance_report.md"
DAILY_CSV = "paper_daily_performance.csv"
IST = ZoneInfo("Asia/Kolkata")


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def close_date_ist(row: dict, fallback: str = "") -> str:
    raw = row.get("closed_at_utc", "")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(IST).date().isoformat()
        except ValueError:
            pass
    return fallback or row.get("date", "")


def summarize(rows):
    n = len(rows)
    wins = [r for r in rows if f(r, "realized_r") > 0]
    losses = [r for r in rows if f(r, "realized_r") < 0]
    breakeven = [r for r in rows if f(r, "realized_r") == 0]
    total_r = sum(f(r, "realized_r") for r in rows)
    pnl = sum(f(r, "realized_pnl") for r in rows)
    gp_r = sum(f(r, "realized_r") for r in wins)
    gl_r = -sum(f(r, "realized_r") for r in losses)
    gp_pnl = sum(f(r, "realized_pnl") for r in wins)
    gl_pnl = -sum(f(r, "realized_pnl") for r in losses)
    return {
        "n": n, "wins": len(wins), "losses": len(losses), "breakeven": len(breakeven),
        "win_rate": len(wins) / n * 100 if n else 0,
        "r": total_r, "pnl": pnl, "avg_r": total_r / n if n else 0,
        "pf_r": gp_r / gl_r if gl_r else float("inf"),
        "pf_pnl": gp_pnl / gl_pnl if gl_pnl else float("inf"),
        "gross_profit_r": gp_r, "gross_loss_r": gl_r,
        "gross_profit_pnl": gp_pnl, "gross_loss_pnl": gl_pnl,
    }


def grouped(rows, key_fn):
    out = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(r)
    return out


def equity_drawdown(rows):
    ordered = sorted(rows, key=lambda r: r.get("closed_at_utc") or r.get("date", ""))
    equity = peak = 0.0
    max_dd = 0.0
    peak_idx = trough_idx = 0
    for i, r in enumerate(ordered, 1):
        equity += f(r, "realized_r")
        if equity > peak:
            peak = equity
            peak_idx = i
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
    return max_dd, peak_idx, trough_idx


def money(x):
    return f"{x:+.5f}"


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)


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

    executed = [r for r in rows if r.get("exit_reason") != "ENTRY_VALIDITY_SKIPPED"]
    skipped = [r for r in rows if r.get("exit_reason") == "ENTRY_VALIDITY_SKIPPED"]
    s = summarize(executed)
    max_dd, peak_trade, trough_trade = equity_drawdown(executed)

    by_asset = grouped(executed, lambda r: r.get("ticker", r.get("asset", "UNKNOWN")))
    by_direction = grouped(executed, lambda r: r.get("direction", "UNKNOWN"))
    by_exit = grouped(executed, lambda r: r.get("exit_reason", "UNKNOWN"))
    by_trailing = grouped(executed, lambda r: round(f(r, "realized_r"), 2) if r.get("exit_reason") == "TRAILING_STOP" else None)
    by_day = defaultdict(list)
    for r in executed:
        by_day[close_date_ist(r)].append(r)
    skipped_day = defaultdict(int)
    for r in skipped:
        skipped_day[close_date_ist(r)] += 1

    daily_rows = []
    cumulative = 0.0
    daily_cumulative = []
    for d in sorted(set(by_day) | set(skipped_day)):
        x = summarize(by_day.get(d, []))
        cumulative += x["r"]
        daily_rows.append((d, x["n"], x["wins"], x["losses"], f'{x["r"]:+.2f}R', money(x["pnl"]), skipped_day[d], f'{cumulative:+.2f}R'))
        daily_cumulative.append((d, cumulative))

    best_r = max(executed, key=lambda r: f(r, "realized_r"), default=None)
    best_pnl = max(executed, key=lambda r: f(r, "realized_pnl"), default=None)
    worst_r = min(executed, key=lambda r: f(r, "realized_r"), default=None)
    worst_pnl = min(executed, key=lambda r: f(r, "realized_pnl"), default=None)

    lines = [
        "# Paper Trading Performance & Strategy Audit",
        "",
        f"**Analysis start:** {cutoff}",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Daily P&L date basis:** trade close date in Asia/Kolkata (IST)",
        "",
        "## Overall Performance",
        f"- Executed trades: **{s['n']}**",
        f"- Skipped signals: **{len(skipped)}**",
        f"- Wins / losses / breakeven: **{s['wins']} / {s['losses']} / {s['breakeven']}**",
        f"- Win rate: **{s['win_rate']:.2f}%**",
        f"- Total realized R: **{s['r']:+.2f}R**",
        f"- Total realized P&L (CSV units): **{money(s['pnl'])}**",
        f"- R profit factor: **{s['pf_r']:.2f}**" if s['gross_loss_r'] else "- R profit factor: **∞**",
        f"- P&L profit factor: **{s['pf_pnl']:.2f}**" if s['gross_loss_pnl'] else "- P&L profit factor: **∞**",
        f"- Average R/trade: **{s['avg_r']:+.3f}R**",
        f"- Maximum trade-sequence drawdown: **{max_dd:.2f}R**",
        "",
        "## Best / Worst Trades",
    ]
    for label, r in [("Best by R", best_r), ("Best by P&L", best_pnl), ("Worst by R", worst_r), ("Worst by P&L", worst_pnl)]:
        if r:
            lines.append(f"- {label}: **{r.get('trade_id','UNKNOWN')}** | {f(r,'realized_r'):+.2f}R | {money(f(r,'realized_pnl'))} | {r.get('exit_reason','')}")

    lines += ["", "## Performance by Asset"]
    asset_table = []
    for a in sorted(by_asset):
        x = summarize(by_asset[a])
        asset_table.append((a, x["n"], x["wins"], x["losses"], f'{x["win_rate"]:.2f}%', f'{x["r"]:+.2f}R', money(x["pnl"]), f'{x["avg_r"]:+.3f}R'))
    table(lines, ["Asset", "Trades", "Wins", "Losses", "Win Rate", "Net R", "Net P&L", "Avg R"], asset_table)

    lines += ["", "## BUY vs SELL"]
    direction_table = []
    for d in sorted(by_direction):
        x = summarize(by_direction[d])
        direction_table.append((d, x["n"], x["wins"], x["losses"], f'{x["win_rate"]:.2f}%', f'{x["r"]:+.2f}R', money(x["pnl"]), f'{x["avg_r"]:+.3f}R'))
    table(lines, ["Direction", "Trades", "Wins", "Losses", "Win Rate", "Net R", "Net P&L", "Avg R"], direction_table)

    lines += ["", "## Performance by Exit Reason"]
    exit_table = []
    for e in sorted(by_exit):
        x = summarize(by_exit[e])
        exit_table.append((e, x["n"], x["wins"], x["losses"], f'{x["r"]:+.2f}R', money(x["pnl"]), f'{x["avg_r"]:+.3f}R'))
    table(lines, ["Exit", "Trades", "Wins", "Losses", "Net R", "Net P&L", "Avg R"], exit_table)

    lines += ["", "## Trailing-System Distribution"]
    trail_table = []
    for level in sorted(k for k in by_trailing if k is not None):
        x = summarize(by_trailing[level])
        trail_table.append((f'{level:+.2f}R', x["n"], f'{x["n"]/s["n"]*100:.2f}%', f'{x["r"]:+.2f}R', money(x["pnl"])))
    table(lines, ["Trailing Level", "Trades", "% Executed", "Total R", "Total P&L"], trail_table)

    lines += ["", "## Daily Performance (IST Close Date)"]
    table(lines, ["Date", "Trades", "Wins", "Losses", "Net R", "Net P&L", "Skipped", "Cumulative R"], daily_rows)

    lines += ["", "## Strategy Audit Signals", ""]
    # These are observations, not automatic strategy changes.
    weakest_asset = min(((a, summarize(v)["r"]) for a, v in by_asset.items()), key=lambda x: x[1], default=("N/A", 0))
    strongest_asset = max(((a, summarize(v)["r"]) for a, v in by_asset.items()), key=lambda x: x[1], default=("N/A", 0))
    weakest_direction = min(((d, summarize(v)["r"]) for d, v in by_direction.items()), key=lambda x: x[1], default=("N/A", 0))
    strongest_direction = max(((d, summarize(v)["r"]) for d, v in by_direction.items()), key=lambda x: x[1], default=("N/A", 0))
    lines += [
        f"- Strongest asset by R: **{strongest_asset[0]} ({strongest_asset[1]:+.2f}R)**",
        f"- Weakest asset by R: **{weakest_asset[0]} ({weakest_asset[1]:+.2f}R)**",
        f"- Strongest direction by R: **{strongest_direction[0]} ({strongest_direction[1]:+.2f}R)**",
        f"- Weakest direction by R: **{weakest_direction[0]} ({weakest_direction[1]:+.2f}R)**",
        "- No asset or direction is automatically disabled by this report.",
        "- No trailing level is automatically changed by this report.",
        "- Use this report as an observation dataset for the one-month audit period before making permanent rule changes.",
        "- Dollar P&L is reported exactly as stored in the CSV; R is the preferred normalized strategy-performance measure.",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(DAILY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date_ist_close", "trades", "wins", "losses", "net_r", "net_pnl", "skipped", "cumulative_r"])
        for row in daily_rows:
            w.writerow(row)

    print("=== PAPER TRADING PERFORMANCE & AUDIT ===")
    print(f"Period: {cutoff} -> latest")
    print(f"Executed: {s['n']} | Skipped: {len(skipped)}")
    print(f"Win rate: {s['win_rate']:.2f}%")
    print(f"Total R: {s['r']:+.2f}R")
    print(f"Total P&L: {money(s['pnl'])}")
    print(f"R profit factor: {s['pf_r']:.2f}" if s['gross_loss_r'] else "R profit factor: inf")
    print(f"Max drawdown: {max_dd:.2f}R")
    print(f"Strongest asset: {strongest_asset[0]} ({strongest_asset[1]:+.2f}R)")
    print(f"Weakest asset: {weakest_asset[0]} ({weakest_asset[1]:+.2f}R)")
    print(f"Strongest direction: {strongest_direction[0]} ({strongest_direction[1]:+.2f}R)")
    print(f"Weakest direction: {weakest_direction[0]} ({weakest_direction[1]:+.2f}R)")


if __name__ == "__main__":
    main()
