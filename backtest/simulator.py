"""
Backtest Trade Simulator - Phase 2 of the approved backtesting framework.

Responsible for resolving already-detected signals into simulated trade
outcomes. Contains zero detection logic of its own.

Reused, unmodified, from main.py: compute_signals(), compute_htf_bias(),
session_mask(). None of BOS/CHoCH detection, ATR, order block selection,
or SL calculation is reimplemented here - see generate_signals_for_backtest()
below, which is a thin orchestration wrapper, not a reimplementation.

ALL exit logic (SL, progressive trailing profit, opposite-signal, the
HIGH-confidence execution gate) is delegated to trade_manager.py, the
single shared trade-management engine also intended for reuse by paper
trading and future live execution. This file contains no exit-decision
logic of its own - see resolve_position_over_bars() below.

Isolation guarantee: this module never imports send_telegram, record_trade,
load_state, or save_state, and never imports from paper_trading.py at all.
It also never imports from data_loader.py - it operates on DataFrames
passed in by the caller, keeping this phase strictly scoped to trade
lifecycle simulation, not data acquisition (that stays Phase 1's job).
"""

from dataclasses import dataclass
from typing import Optional, List

from main import compute_signals, compute_htf_bias, session_mask
from trade_manager import resolve_position_over_bars


@dataclass
class SimulatedTrade:
    signal_index: int
    entry_time: object
    direction: str
    structure: str
    confidence: str
    session_ok: bool
    htf_ok: bool
    entry: float
    sl: float
    tp: float  # retained from the signal dict for reference/backward compatibility only -
               # Change 2 removed fixed-TP as an exit trigger; this value is no longer used
               # anywhere in trade resolution below (see simulate_trades()).
    exit_time: object = None
    exit_price: Optional[float] = None
    exit_reason: str = "STILL_OPEN"  # "SL", "TRAILING_STOP", "OPPOSITE_SIGNAL", "STILL_OPEN"
    r_multiple: Optional[float] = None
    duration_bars: Optional[int] = None
    duration_wall_clock: Optional[object] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None


def _align_htf(df15, df1h, df4h, htf_ema_len):
    """
    Reproduces the HTF-alignment orchestration main.py's process_symbol()
    performs inline. This is NOT a duplication of compute_htf_bias() itself
    (which is imported and called unmodified below) - it's the small amount
    of reindex/ffill/comparison glue around it, which does not exist as a
    standalone importable function in main.py. See Phase 2 self-audit for
    an explicit note on why this couldn't be avoided without modifying
    main.py, which was out of scope for this phase.
    """
    df1h_complete = df1h.iloc[:-1] if len(df1h) > 1 else df1h
    df4h_complete = df4h.iloc[:-1] if len(df4h) > 1 else df4h

    bias1h = compute_htf_bias(df1h_complete, htf_ema_len)
    bias4h = compute_htf_bias(df4h_complete, htf_ema_len)

    bias1h_aligned = bias1h.reindex(df15.index, method="ffill").fillna(0)
    bias4h_aligned = bias4h.reindex(df15.index, method="ffill").fillna(0)

    htf_ok_long = (bias1h_aligned == 1) & (bias4h_aligned == 1)
    htf_ok_short = (bias1h_aligned == -1) & (bias4h_aligned == -1)
    return htf_ok_long, htf_ok_short


def generate_signals_for_backtest(df15, df1h, df4h, use_sessions, htf_ema_len=50):
    """
    Produces the same signal list the live pipeline would produce for this
    data, by calling the same production functions (session_mask,
    compute_htf_bias via _align_htf, compute_signals). No detection logic
    is reimplemented - this function only wires existing pieces together.
    """
    sess_mask = session_mask(df15, use_sessions)
    htf_ok_long, htf_ok_short = _align_htf(df15, df1h, df4h, htf_ema_len)
    signals = compute_signals(df15, sess_mask, htf_ok_long, htf_ok_short)
    return signals


def _track_excursion(df15, entry_index, exit_index, entry, sl, direction):
    """
    MAE/MFE computed over bars entry_index+1 .. exit_index inclusive only.
    Never examines bars after exit_index - this is what guarantees no
    post-exit information leaks into these figures (approved design,
    extended-architecture step, item 1).
    """
    if exit_index <= entry_index:
        return 0.0, 0.0

    window = df15.iloc[entry_index + 1: exit_index + 1]
    risk = abs(entry - sl)
    if risk == 0 or window.empty:
        return 0.0, 0.0

    if direction == "BUY":
        mfe = (window["High"].max() - entry) / risk
        mae = (window["Low"].min() - entry) / risk
    else:
        mfe = (entry - window["Low"].min()) / risk
        mae = (entry - window["High"].max()) / risk

    return mae, mfe


def simulate_trades(df15, signals, all_signals=None) -> List[SimulatedTrade]:
    """
    Walks every signal in chronological order and resolves each one via
    the shared trade_manager engine (SL, progressive trailing profit, or
    the next opposite-direction signal - whichever occurs first). ALL
    exit-decision logic lives in trade_manager.py; this function only
    orchestrates which signals to attempt, and maps the resulting
    Position back onto this file's SimulatedTrade record shape.

    Change 2: a signal whose confidence does not meet the execution
    threshold never opens a trade at all - trade_manager.resolve_position_over_bars()
    returns (None, None) for such signals, and this loop simply skips
    appending anything for them. This applies uniformly regardless of
    which filtered `signals` subset a caller passes in (run_backtest.py's
    ALL/LOW_CONFIDENCE/BOS_ONLY/etc. filters all pass through this same
    gate - the engine enforces it, not the caller).

    Each signal opens its own trade; overlapping/concurrent trades are
    allowed by design. This backtest measures per-signal expectancy (does
    THIS structural break tend to work out), not a single-position-account
    simulation - that constraint belongs to the live state machine
    (a separate, already-approved design), not this research tool.

    `signals` controls which signals are ATTEMPTED (subject to the
    confidence gate above). `all_signals` - the complete, unfiltered
    production signal stream - controls opposite-signal EXIT detection,
    so a trade's exit always matches what the live strategy would
    actually experience, regardless of which subset was used to open it
    (H1 fix, unchanged by this turn). If `all_signals` is not supplied,
    it defaults to `signals`.

    Chronological/no-lookahead guarantee: the outer loop processes signals
    in the order they appear in the `signals` list (already chronological,
    since compute_signals() produces them in bar order). trade_manager's
    resolution loop, for each trade, only ever reads df15 rows from
    entry_index+1 forward - never anything before or at the entry bar's
    own future, and never anything tied to another trade's resolution.
    """
    trades: List[SimulatedTrade] = []

    if all_signals is None:
        all_signals = signals

    for sig in signals:
        entry_index = sig["index"]
        direction = sig["type"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        confidence = sig["confidence"]

        # H1 fix (unchanged): next opposite-direction signal after this
        # one, searched against the COMPLETE production signal stream
        # (all_signals), not just whichever filtered subset opened this
        # trade - so exit timing always matches what the live strategy
        # would actually see. Keyed by bar index, not by position within
        # `signals`, since `all_signals` and `signals` are generally
        # different lists.
        opposite_index = None
        for later_sig in all_signals:
            if later_sig["index"] <= entry_index:
                continue
            if later_sig["type"] != direction:
                opposite_index = later_sig["index"]
                break

        position, exit_index = resolve_position_over_bars(
            direction=direction, entry=entry, sl=sl, confidence=confidence,
            df=df15, entry_index=entry_index, opposite_index=opposite_index,
        )

        if position is None:
            # Change 2: confidence below the execution threshold - never
            # opened, never appears in results, exactly matching "LOW
            # confidence signals must never ... enter backtest execution."
            continue

        trade = SimulatedTrade(
            signal_index=entry_index,
            entry_time=sig["time"],
            direction=direction,
            structure=sig["structure"],
            confidence=confidence,
            session_ok=sig["session_ok"],
            htf_ok=sig["htf_ok"],
            entry=entry,
            sl=sl,
            tp=tp,
        )

        mae, mfe = _track_excursion(df15, entry_index, exit_index, entry, sl, direction)

        trade.exit_time = df15.index[exit_index]
        trade.exit_price = position.exit_price
        trade.exit_reason = position.exit_reason
        trade.r_multiple = position.r_multiple()
        trade.duration_bars = exit_index - entry_index
        trade.duration_wall_clock = df15.index[exit_index] - df15.index[entry_index]
        trade.mae = mae
        trade.mfe = mfe

        trades.append(trade)

    return trades
