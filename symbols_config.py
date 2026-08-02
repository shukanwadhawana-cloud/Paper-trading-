"""
Symbol Configuration - single source of truth for which trading symbols
are supported via the Binance Futures provider, and what Binance ticker
string / contract-type quirks each one needs.

IMPORTANT SCOPE NOTE: this is NOT a replacement for main.py's SYMBOLS
dict or backtest/run_backtest.py's SYMBOLS dict - both are explicitly
frozen, byte-identical strategy-layer files this milestone, governing
Yahoo-based SIGNAL DETECTION, a separate concern from live-data
monitoring. Consolidating those into this file would require editing
protected files, which is out of scope. This file is the single point
of configuration specifically for the Binance-provider code path:
adding a new Binance-supported symbol requires editing ONLY the
dictionary below - no other file in that code path.
"""

# tradfi_perpetual=True marks symbols documented as belonging to
# Binance's newer TradFi Perpetual Contracts framework (see
# market_data.py's BinanceFuturesProvider for the endpoint implications -
# these symbols may require the /fapi/v1/continuousKlines endpoint with
# contractType=TRADIFI_PERPETUAL as a fallback if the standard
# /fapi/v1/klines endpoint doesn't recognize them).
BINANCE_SYMBOLS = {
    "XAUUSDT": {"binance_symbol": "XAUUSDT", "tradfi_perpetual": True},
    "XAGUSDT": {"binance_symbol": "XAGUSDT", "tradfi_perpetual": True},
    "BTCUSDT": {"binance_symbol": "BTCUSDT", "tradfi_perpetual": False},
    "ETHUSDT": {"binance_symbol": "ETHUSDT", "tradfi_perpetual": False},
    "SOLUSDT": {"binance_symbol": "SOLUSDT", "tradfi_perpetual": False},
    "XRPUSDT": {"binance_symbol": "XRPUSDT", "tradfi_perpetual": False},
    "BNBUSDT": {"binance_symbol": "BNBUSDT", "tradfi_perpetual": False},
}


def is_binance_symbol(ticker: str) -> bool:
    ticker = ticker.replace("/", "").replace("-", "").upper()
    return ticker in BINANCE_SYMBOLS


def is_tradfi_perpetual(ticker: str) -> bool:
    ticker = ticker.replace("/", "").replace("-", "").upper()
    entry = BINANCE_SYMBOLS.get(ticker)
    return bool(entry and entry.get("tradfi_perpetual"))