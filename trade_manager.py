"""
Trade Manager - single source of truth for ALL trade exit logic.

Used by:
- backtest/simulator.py (historical simulation)
- main.py (gates whether a signal is executed, via is_executable())
- (future) automated paper-trading position tracking
- (future) Binance / BingX live execution

This module is intentionally symbol- and quote-currency-agnostic: it
operates purely on numeric entry/SL/price values and dimensionless
R-multiples, with no assumption about which asset or which quote
currency (USD, USDT, BTC, ETH, ...) those numbers are denominated in.
Nothing here references a specific symbol, exchange, or currency string.

Isolation: this module has NO dependency on main.py, paper_trading.py,
or backtest/ - it is pure trade-management logic with zero knowledge of
signal detection, data loading, or notification delivery. This keeps it
safely importable from anywhere (research code today, live execution
code later) without circular-import risk.

Does NOT connect to any exchange. exchange_close() and the partial-fill
fields exist as a forward-compatible data model / method surface only -
nothing in this repository calls them yet.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


TRAIL_STEP_R = 0.6

EXIT_SL = "SL"
EXIT_TRAILING_STOP = "TRAILING_STOP"
EXIT_OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"
EXIT_STILL_OPEN = "STILL_OPEN"
EXIT_MANUAL_CLOSE = "MANUAL_CLOSE"      # supported now; not yet called by any part of this repo
EXIT_EXCHANGE_CLOSE = "EXCHANGE_CLOSE"  # supported now; not yet called by any part of this repo

REQUIRE_CONFIDENCE = "HIGH"  # the single point of truth for the confidence-gate rule


def is_executable(confidence: str) -> bool:
    """
    True only if `confidence` meets the execution threshold. This is the
    ONE place the "HIGH confidence only" rule is expressed - callers
    (main.py, backtest/simulator.py, and any future live execution code)
    all call this rather than each re-encoding the rule independently.
    """
    return confidence == REQUIRE_CONFIDENCE


@dataclass
class Position:
    """
    Represents one managed trade, open or closed. Deliberately generic -
    no symbol, no exchange, no hardcoded currency - so the SAME class
    works for Gold, BTC-USD, BTCUSDT, or any future Binance/BingX pair
    without modification.

    `symbol` / `quote_currency` are optional and unused by the current
    backtest path (which already tracks symbol externally, one call per
    ticker) - present here purely so future live-execution code has
    somewhere natural to record that context without needing to change
    this class's shape later.

    `filled_qty` / `partial_fills` are similarly forward-compatible:
    present now, not exercised by any current caller, specifically so a
    future live-execution integration can record partial fills without
    requiring a new Position shape at that point.
    """
    direction: str          # "BUY" or "SELL"
    entry: float
    sl: float
    confidence: str

    symbol: Optional[str] = None
    quote_currency: Optional[str] = None

    locked_level_r: float = field(default=0.0, init=False)
    floor_price: float = field(default=0.0, init=False)
    risk: float = field(default=0.0, init=False)

    is_open: bool = field(default=True, init=False)
    exit_reason: Optional[str] = field(default=None, init=False)
    exit_price: Optional[float] = field(default=None, init=False)

    filled_qty: float = 1.0
    partial_fills: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.risk = abs(self.entry - self.sl)
        self.floor_price = self.sl

    # ---- bar-based updates (used by backtest today) --------------------
    def check_floor_breach(self, bar_high: float, bar_low: float) -> bool:
        """True if this bar's price action breaches the CURRENT floor.
        Callers check this BEFORE calling update_with_bar() for the same
        bar - see resolve_position_over_bars() - so that a single bar's
        favorable excursion can never both raise the floor and be treated
        as surviving its own breach of that new, higher floor."""
        if self.direction == "BUY":
            return bar_low <= self.floor_price
        return bar_high >= self.floor_price

    def update_with_bar(self, bar_high: float, bar_low: float) -> bool:
        """
        Advances the trailing floor by up to one 0.6R step per completed
        level, based on this bar's favorable extreme. "Always lock the
        latest COMPLETED 0.6R level" - floors to the nearest fully-reached
        step (e.g. 0.9R reached locks 0.6R, not 0.9R). Monotonic: a level,
        once locked, is never un-locked even if a later bar's favorable
        excursion is smaller. Returns True if a new level was locked.
        """
        if not self.is_open or self.risk <= 0:
            return False

        if self.direction == "BUY":
            favorable_r = (bar_high - self.entry) / self.risk
        else:
            favorable_r = (self.entry - bar_low) / self.risk

        new_locked_level = math.floor(favorable_r / TRAIL_STEP_R) * TRAIL_STEP_R
        if new_locked_level > self.locked_level_r:
            self.locked_level_r = new_locked_level
            if self.direction == "BUY":
                self.floor_price = self.entry + self.locked_level_r * self.risk
            else:
                self.floor_price = self.entry - self.locked_level_r * self.risk
            return True
        return False

    # ---- single-price updates (forward-compatible for live polling) ----
    def check_price_breach(self, price: float) -> bool:
        """Change 6 forward-compat: a live monitor polling a single last
        price (rather than a completed OHLC bar) can treat that price as
        both the bar high and low for this check. Not called by any
        current path - backtest uses check_floor_breach() with real
        High/Low instead, which is strictly more information."""
        return self.check_floor_breach(price, price)

    def update_with_price(self, price: float) -> bool:
        """Change 6 forward-compat: single-price equivalent of
        update_with_bar(). Not called by any current path."""
        return self.update_with_bar(price, price)

    # ---- closing ---------------------------------------------------
    def close(self, reason: str, price: float):
        self.is_open = False
        self.exit_reason = reason
        self.exit_price = price

    def manual_close(self, price: float):
        """Change 6 forward-compat: human-initiated close. Not called by
        any current path - present so a future manual override doesn't
        require a new close mechanism to be designed later."""
        self.close(EXIT_MANUAL_CLOSE, price)

    def exchange_close(self, price: float):
        """Change 6 forward-compat: exchange-reported close (e.g. a
        liquidation or an exchange-side order fill). Not called by any
        current path - no exchange is connected anywhere in this repo."""
        self.close(EXIT_EXCHANGE_CLOSE, price)

    def apply_partial_fill(self, qty: float, price: float):
        """Change 6 forward-compat: records a partial fill without closing
        the position. Not called by any current path - the backtest and
        live-alert paths in this repo are single-fill only today."""
        self.partial_fills.append({"qty": qty, "price": price})
        self.filled_qty += qty

    def r_multiple(self) -> float:
        if self.risk <= 0 or self.exit_price is None:
            return 0.0
        if self.direction == "BUY":
            return (self.exit_price - self.entry) / self.risk
        return (self.entry - self.exit_price) / self.risk


def resolve_position_over_bars(
    direction: str,
    entry: float,
    sl: float,
    confidence: str,
    tp: Optional[float] = None,
    df,
    entry_index: int,
    opposite_index: Optional[int] = None,
    symbol: Optional[str] = None,
    quote_currency: Optional[str] = None,
) -> Tuple[Optional[Position], Optional[int]]:
    """
    Walks `df` (a DataFrame with High/Low/Close columns) forward from
    entry_index+1, applying SL + progressive trailing-profit exit logic,
    plus an optional opposite-signal exit trigger at `opposite_index`.

    Returns (None, None) if `confidence` does not meet the execution
    threshold (is_executable() is False) - no Position is created at all
    in that case, matching "LOW confidence signals must never open a
    trade." Otherwise returns (a closed-or-STILL_OPEN Position, the bar
    index at which it was resolved).

    This is the single function backtest/simulator.py now delegates to
    instead of containing SL/trailing logic inline.
    """
    if not is_executable(confidence):
        return None, None

    position = Position(
        direction=direction, entry=entry, sl=sl, confidence=confidence,
        symbol=symbol, quote_currency=quote_currency,
    )
    n = len(df)

    for i in range(entry_index + 1, n):
        bar_high = df["High"].iloc[i]
        bar_low = df["Low"].iloc[i]

        # Conservative ordering: this bar's breach of the CURRENT
        # (pre-bar) floor is checked before this bar's favorable
        # excursion is allowed to raise the floor further.
        if position.check_floor_breach(bar_high, bar_low):
            reason = EXIT_SL if position.locked_level_r == 0.0 else EXIT_TRAILING_STOP
            position.close(reason, position.floor_price)
            return position, i

        if opposite_index is not None and i == opposite_index:
            position.close(EXIT_OPPOSITE_SIGNAL, df["Close"].iloc[i])
            return position, i

        position.update_with_bar(bar_high, bar_low)

    last_index = n - 1
    position.close(EXIT_STILL_OPEN, df["Close"].iloc[last_index])
    return position, last_index
