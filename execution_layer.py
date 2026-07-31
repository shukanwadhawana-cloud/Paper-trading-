"""
Execution Layer Abstraction.

Defines the interface through which positions are opened and closed.
PaperExecutionLayer (wrapping paper_trading.py's CSV ledger, unmodified
in its core dedup/idempotency logic) is the only implemented layer
today. BinanceFuturesExecutionLayer and BingXFuturesExecutionLayer are
placeholders reserving the same interface for future live execution.

The intent: moving from paper trading to live execution should require
swapping which ExecutionLayer is active (get_default_execution_layer())
and implementing the two placeholder classes below - nothing about the
strategy (main.py, compute_signals, trade_manager.py) needs to change
for that transition.

Note: main.py's entry point (process_symbol() -> record_trade()) does
NOT yet go through this abstraction - main.py is deliberately unmodified
by this milestone. paper_trade_monitor.py (the new automated monitor)
DOES go through it, for the open-trade-loading and close-trade paths.
Migrating main.py's entry call through this layer is a reasonable future
step, not done here since it wasn't required and main.py needed to stay
untouched.
"""

from abc import ABC, abstractmethod
from typing import List


class ExecutionLayer(ABC):
    @abstractmethod
    def load_open_trades(self) -> List[dict]:
        """Returns all currently-open positions as plain dicts."""
        raise NotImplementedError

    @abstractmethod
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str,
                     realized_r: float, realized_pnl: float,
                     duration_bars: int, duration_wall_clock: str,
                     closed_at_utc: str) -> bool:
        """Closes an open position. Returns False if it was already
        closed (idempotent no-op) - this is what makes duplicate closes
        impossible regardless of how many times this is called for the
        same trade_id."""
        raise NotImplementedError


class PaperExecutionLayer(ExecutionLayer):
    """The only implemented execution layer today. No real orders are
    ever placed - everything routes to paper_trading.py's CSV ledger."""

    def load_open_trades(self) -> List[dict]:
        from paper_trading import load_open_trades as _load_open_trades
        return _load_open_trades()

    def close_trade(self, trade_id, exit_price, exit_reason, realized_r,
                     realized_pnl, duration_bars, duration_wall_clock, closed_at_utc) -> bool:
        from paper_trading import close_trade as _close_trade
        return _close_trade(trade_id, exit_price, exit_reason, realized_r,
                             realized_pnl, duration_bars, duration_wall_clock, closed_at_utc)


class BinanceFuturesExecutionLayer(ExecutionLayer):
    """NOT YET IMPLEMENTED. Reserves the interface for live Binance
    Futures order placement/closing. No exchange connection exists
    anywhere in this repository."""

    def load_open_trades(self) -> List[dict]:
        raise NotImplementedError("Binance Futures execution is not implemented yet.")

    def close_trade(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Binance Futures execution is not implemented yet.")


class BingXFuturesExecutionLayer(ExecutionLayer):
    """NOT YET IMPLEMENTED. Same purpose, for BingX Futures."""

    def load_open_trades(self) -> List[dict]:
        raise NotImplementedError("BingX Futures execution is not implemented yet.")

    def close_trade(self, *args, **kwargs) -> bool:
        raise NotImplementedError("BingX Futures execution is not implemented yet.")


def get_default_execution_layer() -> ExecutionLayer:
    """Single point of configuration - swap to a live layer here once
    implemented; nothing else in paper_trade_monitor.py needs to change."""
    return PaperExecutionLayer()
