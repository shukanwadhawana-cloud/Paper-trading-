"""
Gold/Silver SMC Signal Bot
Structure-detection logic ported from LuxAlgo's "Smart Money Concepts" indicator
(CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/, (c) LuxAlgo)
Ported to Python for personal, non-commercial alerting use.

- Pulls 15m candles for Gold (GC=F) and Silver (SI=F) via yfinance
- Detects swing structure via a lag-based "leg" detector (same method as the
  original indicator), then flags BOS / CHoCH on close crossing the last
  unbroken swing pivot
- Order blocks are found by scanning the whole leg range for the most
  extreme ATR-volatility-adjusted candle (matches LuxAlgo's storeOrdeBlock)
- Filters by Asia/London/US session and 1H+4H trend alignment
- Sends a Telegram message only when a NEW signal appears on the most
  recently CLOSED 15m candle (de-duplicated via a state file committed
  back to the repo)
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from paper_trading import record_trade
from trade_manager import is_executable

# ================= CONFIG =================
# SYMBOLS: each entry controls whether the Asia/London/US session filter applies.
# Gold/Silver (futures, near-24h but with real weekly closes) benefit from session filtering.
# Bitcoin/Ethereum trade 24/7/365 - forcing them through forex sessions would suppress
# valid signals outside those windows, so session filtering is off for them by default.
SYMBOLS = {
    "Gold (XAU/USD)":     {"ticker": "GC=F",    "use_sessions": True},
    "Silver (XAG/USD)":   {"ticker": "SI=F",    "use_sessions": True},
    "Bitcoin (BTC/USD)":  {"ticker": "BTC-USD", "use_sessions": False},
    "Ethereum (ETH/USD)": {"ticker": "ETH-USD", "use_sessions": False},
}

RR_RATIO       = 4.0     # 1:4. Change to 5.0 for 1:5, etc.
STRUCTURE_SIZE = 5       # leg lookback bars (LuxAlgo "internal" tier default = 5, "swing" tier default = 50)
ATR_LEN        = 200     # matches ta.atr(200) in the original indicator
SL_BUFFER_PCT  = 0.0005  # 0.05% buffer beyond the order block for stop loss
# NOTE: session/HTF checks below no longer block alerts - every BOS/CHoCH break sends a
# message. The checks are shown IN the message (✅/❌) so you can judge confidence yourself.
REQUIRE_HTF    = True    # kept as the label for whether HTF is checked/shown at all
HTF_EMA_LEN    = 50

USE_ASIA   = True
USE_LONDON = True
USE_US     = True
ASIA_START, ASIA_END     = 0, 9
LONDON_START, LONDON_END = 7, 16
US_START, US_END         = 12, 24   # extended to midnight UTC to close a 21:00-24:00 UTC gap where no session applied

STATE_FILE = "last_signals.json"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")


# ================= HELPERS =================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns like ('Close', 'GC=F').
    Flatten to plain 'Close', 'Open', etc. so downstream code works either way."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def compute_atr(df: pd.DataFrame, length: int = 200) -> pd.Series:
    """Wilder's ATR - matches Pine's ta.atr(length)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def compute_parsed_hl(df: pd.DataFrame, atr: pd.Series):
    """Matches LuxAlgo's parsedHigh/parsedLow: on abnormally wide-range bars,
    swap high/low so a single huge wick doesn't distort order block detection."""
    rng = df["High"] - df["Low"]
    high_vol = rng >= (2 * atr)
    parsed_high = np.where(high_vol, df["Low"], df["High"])
    parsed_low = np.where(high_vol, df["High"], df["Low"])
    return pd.Series(parsed_high, index=df.index), pd.Series(parsed_low, index=df.index)


def compute_leg_structure(df: pd.DataFrame, size: int):
    """
    Port of LuxAlgo's leg() + getCurrentStructure() + displayStructure().
    Returns a chronological list of structure break events:
      {'index', 'time', 'type': 'BOS'/'CHOCH', 'bias': 'BULLISH'/'BEARISH',
       'pivot_bar': int, 'pivot_level': float}
    """
    n = len(df)
    highest = df["High"].rolling(size).max()
    lowest = df["Low"].rolling(size).min()
    high_shift = df["High"].shift(size)
    low_shift = df["Low"].shift(size)

    leg = 0
    swing_high = {"level": None, "crossed": False, "bar": None}
    swing_low = {"level": None, "crossed": False, "bar": None}
    trend_bias = 0  # 0 neutral, 1 bullish, -1 bearish

    events = []

    for i in range(n):
        prev_leg = leg

        if i >= size and not np.isnan(highest.iloc[i]) and not np.isnan(lowest.iloc[i]):
            new_leg_high = high_shift.iloc[i] > highest.iloc[i]
            new_leg_low = low_shift.iloc[i] < lowest.iloc[i]
            if new_leg_high:
                leg = 0  # BEARISH_LEG
            elif new_leg_low:
                leg = 1  # BULLISH_LEG

        new_pivot = (leg != prev_leg) and i > 0

        if new_pivot:
            if leg == 1:  # start of a bullish leg -> a new swing LOW just formed
                swing_low = {"level": df["Low"].iloc[i - size], "crossed": False, "bar": i - size}
            else:  # start of a bearish leg -> a new swing HIGH just formed
                swing_high = {"level": df["High"].iloc[i - size], "crossed": False, "bar": i - size}

        close_i = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1] if i > 0 else np.nan

        if swing_high["level"] is not None and not swing_high["crossed"]:
            if not np.isnan(prev_close) and prev_close <= swing_high["level"] and close_i > swing_high["level"]:
                tag = "CHOCH" if trend_bias == -1 else "BOS"
                swing_high["crossed"] = True
                trend_bias = 1
                events.append({
                    "index": i, "time": df.index[i], "type": tag, "bias": "BULLISH",
                    "pivot_bar": swing_high["bar"], "pivot_level": swing_high["level"],
                })

        if swing_low["level"] is not None and not swing_low["crossed"]:
            if not np.isnan(prev_close) and prev_close >= swing_low["level"] and close_i < swing_low["level"]:
                tag = "CHOCH" if trend_bias == 1 else "BOS"
                swing_low["crossed"] = True
                trend_bias = -1
                events.append({
                    "index": i, "time": df.index[i], "type": tag, "bias": "BEARISH",
                    "pivot_bar": swing_low["bar"], "pivot_level": swing_low["level"],
                })

    return events


def find_order_block(parsed_high: pd.Series, parsed_low: pd.Series, pivot_bar, break_bar, bias):
    """Port of storeOrdeBlock: scans the whole leg range for the most extreme
    volatility-adjusted candle. bias here matches the *break* direction
    (BULLISH break -> look for the deepest low; BEARISH break -> highest high)."""
    if pivot_bar is None:
        return None
    lo = max(pivot_bar, 0)
    hi = break_bar
    if hi <= lo:
        return None

    if bias == "BEARISH":
        window = parsed_high.iloc[lo:hi]
        if window.empty:
            return None
        idx = window.idxmax()
    else:
        window = parsed_low.iloc[lo:hi]
        if window.empty:
            return None
        idx = window.idxmin()

    return {"top": parsed_high.loc[idx], "bottom": parsed_low.loc[idx]}


def compute_htf_bias(df_htf: pd.DataFrame, ema_len: int = 50) -> pd.Series:
    df_htf = df_htf.copy()
    df_htf["ema"] = df_htf["Close"].ewm(span=ema_len, adjust=False).mean()
    bullish = (df_htf["Close"] > df_htf["ema"]) & (df_htf["Close"] > df_htf["Open"])
    bearish = (df_htf["Close"] < df_htf["ema"]) & (df_htf["Close"] < df_htf["Open"])
    bias = pd.Series(0, index=df_htf.index)
    bias[bullish] = 1
    bias[bearish] = -1
    return bias


def session_mask(df: pd.DataFrame, use_sessions: bool = True) -> pd.Series:
    if not use_sessions:
        return pd.Series(True, index=df.index)

    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    hours = idx.hour

    asia = (hours >= ASIA_START) & (hours < ASIA_END)
    london = (hours >= LONDON_START) & (hours < LONDON_END)
    us = (hours >= US_START) & (hours < US_END)

    mask = pd.Series(False, index=df.index)
    if USE_ASIA:
        mask |= asia
    if USE_LONDON:
        mask |= london
    if USE_US:
        mask |= us
    return mask


def compute_signals(df, sess_mask, htf_ok_long, htf_ok_short):
    """Every BOS/CHoCH break produces a signal immediately - no waiting for a
    retest. Session and HTF alignment are no longer gates; they're recorded
    on the signal so the alert can show ✅/❌ and a HIGH/LOW confidence tag."""
    atr = compute_atr(df, ATR_LEN)
    parsed_high, parsed_low = compute_parsed_hl(df, atr)
    events = compute_leg_structure(df, STRUCTURE_SIZE)

    signals = []

    for ev in events:
        i = ev["index"]
        close_i = df["Close"].iloc[i]
        sess_ok = bool(sess_mask.iloc[i])

        if ev["bias"] == "BULLISH":
            ob = find_order_block(parsed_high, parsed_low, ev["pivot_bar"], ev["index"], "BULLISH")
            sl_anchor = ob["bottom"] if ob else ev["pivot_level"]
            htf_ok = bool(htf_ok_long.iloc[i])
            sl = sl_anchor * (1 - SL_BUFFER_PCT)
            risk = close_i - sl
            if risk <= 0:
                continue
            tp = close_i + risk * RR_RATIO
            signals.append({
                "index": i, "time": df.index[i], "type": "BUY",
                "entry": close_i, "sl": sl, "tp": tp, "structure": ev["type"],
                "session_ok": sess_ok, "htf_ok": htf_ok,
                "confidence": "HIGH" if (sess_ok and htf_ok) else "LOW",
            })
        else:
            ob = find_order_block(parsed_high, parsed_low, ev["pivot_bar"], ev["index"], "BEARISH")
            sl_anchor = ob["top"] if ob else ev["pivot_level"]
            htf_ok = bool(htf_ok_short.iloc[i])
            sl = sl_anchor * (1 + SL_BUFFER_PCT)
            risk = sl - close_i
            if risk <= 0:
                continue
            tp = close_i - risk * RR_RATIO
            signals.append({
                "index": i, "time": df.index[i], "type": "SELL",
                "entry": close_i, "sl": sl, "tp": tp, "structure": ev["type"],
                "session_ok": sess_ok, "htf_ok": htf_ok,
                "confidence": "HIGH" if (sess_ok and htf_ok) else "LOW",
            })

    return signals


def debug_last_event(df, sess_mask, htf_ok_long, htf_ok_short):
    """When nothing fires, explain why - was there no structure break at all,
    or was one found but blocked by session/HTF filters?"""
    events = compute_leg_structure(df, STRUCTURE_SIZE)
    if not events:
        return "no structure break (BOS/CHoCH) detected in this window at all."
    ev = events[-1]
    i = ev["index"]
    sess_ok = bool(sess_mask.iloc[i])
    htf_ok = bool(htf_ok_long.iloc[i]) if ev["bias"] == "BULLISH" else bool(htf_ok_short.iloc[i])
    bars_ago = (len(df) - 1) - i
    return (f"last structure break was {ev['type']} {ev['bias']} at {ev['time']} "
            f"({bars_ago} bars ago) - session_ok={sess_ok}, htf_ok={htf_ok}")


def send_telegram(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID - cannot send. Check repo secrets.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except requests.RequestException as exc:
        print(f"Telegram send failed (network error): {exc}")
        return False
    if resp.status_code != 200:
        print("Telegram send failed:", resp.text)
        return False
    return True


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # C2 fix: a corrupted/truncated state file (e.g. from a runner
            # killed mid-write) must never crash every subsequent scheduled
            # run. Recover with empty state instead - worst case this
            # re-sends a signal that was already sent once, which is far
            # safer than a self-perpetuating total pipeline failure.
            print(f"WARNING: {STATE_FILE} is corrupted or unreadable ({exc}). Starting with empty state.")
            return {}
    return {}


def save_state(state):
    # C2 fix: write to a temp file in the same directory, then atomically
    # replace the real file. os.replace() is atomic on both POSIX and
    # Windows, so a killed/interrupted run can never leave last_signals.json
    # partially written - the file is always either the old complete
    # version or the new complete version, never something in between.
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE)


def process_symbol(label, cfg, state):
    ticker = cfg["ticker"]
    use_sessions = cfg["use_sessions"]

    df15 = yf.download(ticker, period="5d", interval="15m", progress=False)
    df1h = yf.download(ticker, period="60d", interval="60m", progress=False)

    if df15.empty or df1h.empty or len(df15) < STRUCTURE_SIZE * 2 + 2:
        print(f"{label}: not enough data this run, skipping.")
        return

    df15 = flatten_columns(df15)
    df1h = flatten_columns(df1h)

    df4h = df1h.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    df1h_complete = df1h.iloc[:-1] if len(df1h) > 1 else df1h
    df4h_complete = df4h.iloc[:-1] if len(df4h) > 1 else df4h

    bias1h = compute_htf_bias(df1h_complete, HTF_EMA_LEN)
    bias4h = compute_htf_bias(df4h_complete, HTF_EMA_LEN)

    bias1h_aligned = bias1h.reindex(df15.index, method="ffill").fillna(0)
    bias4h_aligned = bias4h.reindex(df15.index, method="ffill").fillna(0)

    if REQUIRE_HTF:
        htf_ok_long = (bias1h_aligned == 1) & (bias4h_aligned == 1)
        htf_ok_short = (bias1h_aligned == -1) & (bias4h_aligned == -1)
    else:
        htf_ok_long = pd.Series(True, index=df15.index)
        htf_ok_short = pd.Series(True, index=df15.index)

    sess_mask = session_mask(df15, use_sessions)

    signals = compute_signals(df15, sess_mask, htf_ok_long, htf_ok_short)

    last_closed_idx = len(df15) - 2
    FRESHNESS_BARS = 4  # tolerate up to ~1 hour of scheduler delay before a signal is considered too stale to alert on

    candidates = [s for s in signals if last_closed_idx - FRESHNESS_BARS <= s["index"] <= last_closed_idx]

    if not candidates:
        dbg = debug_last_event(df15, sess_mask, htf_ok_long, htf_ok_short)
        print(f"{label}: no signals in current window. {dbg}")
        return

    latest = candidates[-1]

    # Change 1/5: the "HIGH confidence only" rule is now expressed in
    # exactly one place - trade_manager.is_executable() - reused
    # identically by backtest/simulator.py. LOW confidence signals are
    # logged for analysis only - never sent, never recorded as a trade.
    # This does not touch candidate selection (candidates[-1]) or
    # compute_signals() - both unchanged.
    if not is_executable(latest["confidence"]):
        print(
            f"{label}: LOW confidence signal detected ({latest['type']} {latest['structure']} "
            f"at {latest['time']}) - not executed per strategy rules (HIGH confidence only). Logged only."
        )
        return

    sig_key = f"{ticker}_{latest['time'].isoformat()}_{latest['type']}"
    if state.get(ticker) == sig_key:
        print(f"{label}: signal already sent, skipping.")
        return

    # Phase 1: record this as a virtual trade regardless of Telegram outcome below -
    # paper trade history must never depend on notification delivery succeeding.
    trade_id = record_trade(ticker, label, latest)

    star = "⭐⭐ HIGH CONFIDENCE\n" if latest["confidence"] == "HIGH" else "Lower Confidence\n"
    htf_mark = "✅" if latest["htf_ok"] else "❌"
    sess_mark = "✅" if latest["session_ok"] else "❌"

    msg = (
        f"{latest['type']} signal — {label}\n"
        f"Trade ID: {trade_id}\n"
        f"{star}"
        f"Structure: {latest['structure']}\n"
        f"HTF Alignment: {htf_mark}\n"
        f"Session: {sess_mark}\n"
        f"Time: {latest['time']}\n"
        f"Entry: {latest['entry']:.2f}\n"
        f"SL: {latest['sl']:.2f}\n"
        f"TP: {latest['tp']:.2f}\n"
        f"RR: 1:{RR_RATIO:g}"
    )
    sent_ok = send_telegram(msg)
    print(msg)
    if sent_ok:
        state[ticker] = sig_key
    else:
        print(f"{label}: WARNING - signal was found but Telegram send failed. Will retry next run.")


def main():
    state = load_state()

    for label, cfg in SYMBOLS.items():
        try:
            process_symbol(label, cfg, state)
        except Exception as exc:
            # One symbol failing (bad data, rate limit, etc.) must never take down the others.
            print(f"{label}: ERROR - {exc}")

    save_state(state)


if __name__ == "__main__":
    main()
