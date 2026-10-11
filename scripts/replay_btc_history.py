#!/usr/bin/env python3
"""Recompute historical BTC paper-trade outcomes from completed BingX swap candles."""
from __future__ import annotations
import argparse, csv, os
from collections import Counter
from datetime import datetime, timezone
import ccxt
import pandas as pd
from trade_manager import resolve_position_over_bars, EXIT_STILL_OPEN

INTERVAL = "15m"
DELTA = pd.Timedelta(minutes=15)
SYMBOL = "BTC/USDT:USDT"

def parse_utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def fetch_history(start):
    exchange = ccxt.bingx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(start.timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    collected = {}
    for _ in range(80):
        batch = exchange.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, since=since, limit=500)
        if not batch:
            break
        for candle in batch:
            collected[int(candle[0])] = candle
        last_ms = int(batch[-1][0])
        if last_ms < since or last_ms >= now_ms:
            break
        # Some exchanges return fewer than the requested limit even when more
        # history exists. Continue pagination until the endpoint is empty or current.
        next_since = last_ms + 1
        if next_since <= since:
            break
        since = next_since
    if not collected:
        raise RuntimeError("BingX returned no BTC swap candles; replay is not verified.")
    times = sorted(collected)
    df = pd.DataFrame(
        [{"Open": float(collected[t][1]), "High": float(collected[t][2]),
          "Low": float(collected[t][3]), "Close": float(collected[t][4])} for t in times],
        index=pd.to_datetime(times, unit="ms", utc=True))
    df.index.name = "open_time"
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="paper_trades.csv")
    parser.add_argument("--out-dir", default="replay-output")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.ledger, newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    btc_rows = [r for r in all_rows if "BTC" in (r.get("ticker") or "").upper() and r.get("entry_time_utc")]
    if not btc_rows:
        raise RuntimeError("No BTC ledger rows with entry timestamps found.")
    bars = fetch_history(min(parse_utc(r["entry_time_utc"]) + DELTA for r in btc_rows))
    now = pd.Timestamp.now(tz="UTC")
    results = []
    for row in btc_rows:
        status, old_reason, old_r = (row.get("status") or "").upper(), row.get("exit_reason") or "", row.get("realized_r") or ""
        result = {k: row.get(k, "") for k in ("trade_id","entry_time_utc","direction","entry","sl","tp","exit_price","closed_at_utc")}
        result["ledger_exit_price"] = result.pop("exit_price")
        result["ledger_closed_at_utc"] = result.pop("closed_at_utc")
        result.update(ledger_status=status, ledger_exit_reason=old_reason, ledger_realized_r=old_r,
                      replay_status="", replay_exit_reason="", replay_exit_price="", replay_exit_time_utc="",
                      replay_realized_r="", r_delta_vs_ledger="", replay_bars="", comparison="", notes="")
        if old_reason == "ENTRY_VALIDITY_SKIPPED":
            result.update(replay_status="SKIPPED", replay_exit_reason=old_reason, replay_realized_r="0.00000",
                          comparison="MATCH", notes="Kept skipped; not counted as an executed trade.")
            results.append(result)
            continue
        entry_time = parse_utc(row["entry_time_utc"]) + DELTA
        post = bars.loc[(bars.index >= entry_time) & ((bars.index + DELTA) <= now)]
        if post.empty:
            result.update(replay_status="UNVERIFIABLE", comparison="UNVERIFIABLE",
                          notes="No completed post-entry candles available.")
            results.append(result)
            continue
        position, exit_idx = resolve_position_over_bars(
            direction=row["direction"], entry=float(row["entry"]), sl=float(row["sl"]),
            confidence=row.get("confidence", ""), df=post, entry_index=-1, opposite_index=None,
            symbol="BTCUSDT", quote_currency="USDT", tp=float(row["tp"]) if row.get("tp") else None)
        if position is None:
            result.update(replay_status="UNVERIFIABLE", comparison="UNVERIFIABLE",
                          notes="Trade failed current executable-confidence rule.")
            results.append(result)
            continue
        result["replay_bars"] = str(exit_idx + 1 if exit_idx is not None else len(post))
        if position.exit_reason == EXIT_STILL_OPEN or position.exit_price is None:
            result.update(replay_status="OPEN_AT_REPLAY_END", replay_exit_reason="STILL_OPEN",
                          comparison="UNVERIFIABLE", notes="No exit reached in available completed candles.")
            results.append(result)
            continue
        replay_r = position.r_multiple()
        result.update(replay_status="CLOSED", replay_exit_reason=position.exit_reason,
                      replay_exit_price=f"{position.exit_price:.5f}",
                      replay_exit_time_utc=(post.index[exit_idx] + DELTA).isoformat(),
                      replay_realized_r=f"{replay_r:.5f}")
        if status != "CLOSED":
            result["comparison"] = "NOT_COMPARABLE_OPEN_LEDGER"
        else:
            try:
                result["r_delta_vs_ledger"] = f"{replay_r-float(old_r):+.5f}"
                result["comparison"] = "MATCH" if abs(float(old_r)-replay_r) < 0.00001 and old_reason == position.exit_reason and abs(float(row.get("exit_price") or 0)-float(position.exit_price)) < 0.00001 else "MISMATCH"
            except (ValueError, TypeError):
                result["comparison"] = "MISMATCH"
        results.append(result)
    output_csv = os.path.join(args.out_dir, "btc_replay_comparison.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    executed = [r for r in results if r["replay_status"] == "CLOSED" and r["replay_exit_reason"] != "ENTRY_VALIDITY_SKIPPED"]
    values = [float(r["replay_realized_r"]) for r in executed]
    wins, losses, total_r = sum(v > 0 for v in values), sum(v < 0 for v in values), sum(values)
    counts = Counter(r["comparison"] for r in results)
    mismatches = [r for r in results if r["comparison"] == "MISMATCH"]
    ledger_executed = [r for r in btc_rows if (r.get("status") or "").upper() == "CLOSED" and (r.get("exit_reason") or "") != "ENTRY_VALIDITY_SKIPPED"]
    ledger_total_r = sum(float(r.get("realized_r") or 0) for r in ledger_executed)
    ledger_wins = sum(float(r.get("realized_r") or 0) > 0 for r in ledger_executed)
    ledger_losses = sum(float(r.get("realized_r") or 0) < 0 for r in ledger_executed)
    unverifiable = [r for r in results if r["comparison"] in ("UNVERIFIABLE", "NOT_COMPARABLE_OPEN_LEDGER")]
    report = [
        "# BTC Historical Replay Audit", "",
        f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Source ledger rows: {len(btc_rows)} BTC records",
        f"- Candle source: BingX USDT-margined swap, 15-minute OHLCV ({SYMBOL})",
        "- Method: exclude signal candle; use completed post-entry candles only; apply current trade-manager exit rules.",
        "- Provenance limitation: the historical ledger does not store the exchange/provider used for each original exit. The current canonical replay uses BingX for every BTC candle; discrepancies cannot be conclusively attributed to one cause per row.",
        "- Resolution rule: preserve the original ledger unchanged; use this same-source replay as the canonical comparison series and retain original values for auditability. Do not silently overwrite history.",
        "- Limits: this replay does not establish actual fills, fees, slippage, funding, or live profitability.",
        "", "## Original ledger baseline (not independently source-verified)",
        f"- Closed executed trades in ledger: {len(ledger_executed)}",
        f"- Ledger wins/losses: {ledger_wins}/{ledger_losses}",
        f"- Ledger net: {ledger_total_r:+.2f}R",
        "", "## Replayed outcomes",
        f"- Replayed closed executed trades: {len(executed)}", f"- Wins: {wins}", f"- Losses: {losses}",
        f"- Net R: {total_r:+.2f}R",
        f"- Replay minus ledger net: {total_r-ledger_total_r:+.2f}R",
        f"- Average R per replayed closed trade: {total_r/len(executed) if executed else 0:+.3f}R",
        f"- Ledger comparison counts: {dict(counts)}", f"- Mismatches requiring review: {len(mismatches)}",
        f"- Unverifiable/open comparisons: {len(unverifiable)}", "",
        "## Interpretation",
        "- A positive R total is not monetary profit and does not by itself prove strategy edge.",
        "- Any incomplete candle coverage or unverifiable trades makes the result provisional.",
        "- Inspect the row-level CSV before declaring the historical ledger reconciled.",
        "", "## Mismatches",
        *[f"- {r['trade_id']}: ledger {r['ledger_exit_reason']} at {r['ledger_exit_price']} ({r['ledger_realized_r']}R, {r['ledger_closed_at_utc']}) vs BingX replay {r['replay_exit_reason']} at {r['replay_exit_price']} ({r['replay_realized_r']}R, {r['replay_exit_time_utc']}); delta {r['r_delta_vs_ledger']}R" for r in mismatches],
        "", "## Unverifiable rows",
        *[f"- {r['trade_id']}: {r['notes']}" for r in results if r['comparison'] == "UNVERIFIABLE"]
    ]
    with open(os.path.join(args.out_dir, "btc_replay_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    if not executed:
        raise RuntimeError("No closed executed trades replayed; performance conclusion is not possible.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
