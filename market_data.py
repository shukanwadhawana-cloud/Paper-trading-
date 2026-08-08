"""
Market Data Abstraction Layer.

Defines a provider-agnostic interface for fetching OHLC price bars and
current price, so the Paper Trade Monitor (and, later, live execution)
never depends directly on any single data source. Yahoo Finance and
Binance Futures are both implemented; BingX Futures remains a reserved
placeholder.

Nothing in this file contains trading/exit logic - it only fetches and
normalizes market data into a consistent shape (a DataFrame with
Open/High/Low/Close columns, indexed by UTC-aware timestamps, or a plain
float for current price).

This file is completely separate from backtest/data_loader.py (Phase 1
of the backtesting framework, unmodified, untouched by this milestone).
That module exists specifically for historical-research data fetching;
this one exists specifically for the live Paper Trade Monitor's needs.

BINANCE ENDPOINT VERIFICATION NOTE (per explicit instruction to verify
before implementing): Binance's own USDS-margined-futures API changelog
documents a "TRADIFI_PERPETUAL" contract type on the
/fapi/v1/continuousKlines endpoint, distinct from the standard
symbol-based /fapi/v1/klines endpoint used for ordinary perpetuals
(BTCUSDT, ETHUSDT, etc.). This strongly suggests XAUUSDT/XAGUSDT (which
Binance documents as belonging to its "TradFi Perpetual" framework, live
since January 2026) may require this different endpoint. No live network
access was available to empirically confirm which path actually succeeds
for these two symbols specifically, so BinanceFuturesProvider.get_bars()
tries the standard endpoint first and automatically falls back to
continuousKlines+TRADIFI_PERPETUAL only for the two symbols documented as
needing it (see symbols_config.is_tradfi_perpetual()). This should be
verified against the real live endpoint once deployed with real network
access - if the standard endpoint turns out to work directly for these
two symbols, the fallback path simply never triggers and this remains
correct; if it doesn't, the fallback handles it automatically either way.

PROVIDER ROUTING FIX (this revision): get_provider_for_ticker() and
get_default_provider() previously routed based on either an exact-match
dict lookup (fragile against ticker format variations like "ETH/USDT"
vs "ETHUSDT") or, in a later edit, were hardcoded to unconditionally
return BingXFuturesProvider() - an unimplemented placeholder that raises
NotImplementedError for every call. Both were bugs. Routing is now based
on a normalized (separator/case-insensitive) comparison against
symbols_config.BINANCE_SYMBOLS, with a pattern-based fallback for
Binance-style USDT-margined tickers not yet explicitly registered there,
and never selects BingXFuturesProvider. See _normalize_ticker(),
_is_binance_ticker(), _is_yahoo_ticker() below.
"""

import re
from abc import ABC, abstractmethod
import pandas as pd

from symbols_config import is_tradfi_perpetual


class MarketDataProvider(ABC):
    """Base interface every market-data provider must implement."""

    @abstractmethod
    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [Open, High, Low, Close], indexed
        by UTC-aware timestamps, for `symbol` at the given `interval`.

        Either `start` (fetch bars from this timestamp forward) or `period`
        (a provider-specific lookback string, e.g. "5d") may be supplied by
        the caller. Providers may support one, both, or translate between
        them as needed.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Returns the latest traded price for `symbol` as a plain float."""
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """
    Wraps yfinance. Uses main.py's flatten_columns() (imported, not
    duplicated) to normalize yfinance's occasional MultiIndex-column
    output - the same fix already relied on by main.py and
    backtest/data_loader.py.
    """

    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        import yfinance as yf
        from main import flatten_columns

        kwargs = {"interval": interval, "progress": False}
        if start is not None:
            kwargs["start"] = start
        if period is not None:
            kwargs["period"] = period
        elif start is None:
            kwargs["period"] = "5d"

        df = yf.download(symbol, **kwargs)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

        df = flatten_columns(df)
        return df[["Open", "High", "Low", "Close"]]

    def get_current_price(self, symbol: str) -> float:
        """Reuses get_bars() rather than introducing a separate,
        untested yfinance API surface - the latest 1-minute bar's close
        is a reasonable proxy for current price."""
        df = self.get_bars(symbol, interval="1m", period="1d")
        if df.empty:
            raise ValueError(f"YahooFinanceProvider: no recent bars available for {symbol}")
        return float(df["Close"].iloc[-1])


def _parse_klines(raw: list) -> pd.DataFrame:
    """Shared parser for both /fapi/v1/klines and /fapi/v1/continuousKlines -
    Binance documents an identical array-of-arrays response shape for both:
    [open_time_ms, open, high, low, close, volume, close_time_ms, ...]."""
    if not raw:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    times, rows = [], []
    for k in raw:
        times.append(pd.Timestamp(int(k[0]), unit="ms", tz="UTC"))
        rows.append({"Open": float(k[1]), "High": float(k[2]), "Low": float(k[3]), "Close": float(k[4])})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name="open_time"))


def _parse_period_to_timedelta(period: str) -> pd.Timedelta:
    """Minimal parser covering the simple 'Nd'/'Nh'/'Nm' strings actually
    used anywhere in this repository (e.g. '5d'). Not a general-purpose
    parser - kept intentionally small and scoped to real call patterns."""
    match = re.match(r"^(\d+)([dhm])$", period.strip())
    if not match:
        raise ValueError(f"Unsupported period format: {period!r}")
    n, unit = int(match.group(1)), match.group(2)
    unit_map = {"d": "days", "h": "hours", "m": "minutes"}
    return pd.Timedelta(**{unit_map[unit]: n})


class BinanceFuturesProvider(MarketDataProvider):
    """
    Binance USDS-margined Futures public market data. Uses ONLY public
    REST endpoints - no API key is required or used for market data
    (klines and ticker/price are both public Binance endpoints). No
    account access, no order placement, no authentication of any kind
    exists anywhere in this class.
    """

    BASE_URL = "https://fapi.binance.com"
    KLINES_LIMIT = 1500  # Binance's documented max per call for USDS-margined futures klines
    MAX_PAGES = 20        # safety valve: 20 x 1500 x 15min ~= 312 days - far beyond any realistic open-trade duration

    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        import requests

        start_ts = None
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
        elif period is not None:
            start_ts = pd.Timestamp.now(tz="UTC") - _parse_period_to_timedelta(period)

        try:
            return self._fetch_paginated(
                f"{self.BASE_URL}/fapi/v1/klines",
                {"symbol": symbol, "interval": interval}, start_ts,
            )
        except requests.RequestException:
            if is_tradfi_perpetual(symbol):
                # See module docstring: documented fallback for Binance's
                # TradFi Perpetual framework (XAUUSDT/XAGUSDT). Paginated
                # identically to the primary path above.
                return self._fetch_paginated(
                    f"{self.BASE_URL}/fapi/v1/continuousKlines",
                    {"pair": symbol, "contractType": "TRADIFI_PERPETUAL", "interval": interval}, start_ts,
                )
            raise

    def _fetch_paginated(self, url: str, base_params: dict, start_ts) -> pd.DataFrame:
        """
        Shared pagination loop for both /fapi/v1/klines and
        /fapi/v1/continuousKlines (identical response shape and paging
        semantics per Binance's documentation). Advances startTime by
        exactly (last_bar_open_time + 1ms) each page, guaranteeing forward
        progress regardless of data content - this is what makes the loop
        provably terminate rather than relying on a specific row count.
        MAX_PAGES is a defensive cap, not an expected limit under normal use.
        """
        import requests

        all_rows = []
        current_start = start_ts
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

        for _ in range(self.MAX_PAGES):
            params = dict(base_params)
            params["limit"] = self.KLINES_LIMIT
            if current_start is not None:
                params["startTime"] = int(current_start.timestamp() * 1000)

            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            page = resp.json()

            if not page:
                break

            all_rows.extend(page)
            last_open_ms = int(page[-1][0])

            # Stop once this page's last bar is already at/after "now" -
            # no more data can exist beyond that point.
            if last_open_ms >= now_ms or len(page) < self.KLINES_LIMIT:
                break

            current_start = pd.Timestamp(last_open_ms + 1, unit="ms", tz="UTC")

        return _parse_klines(all_rows)

    def get_current_price(self, symbol: str) -> float:
        import requests

        symbol = symbol.replace("/", "")

        resp = requests.get(
            f"{self.BASE_URL}/fapi/v1/ticker/price",
        params={"symbol": symbol},
        timeout=10,
        )
        resp.raise_for_status()

        data = resp.json()

        if "price" not in data:
            raise ValueError(
                f"BinanceProvider: unexpected ticker response for {symbol}: {data}"
            )

    return float(data["price"])

class BingXFuturesProvider(MarketDataProvider):
    """NOT YET IMPLEMENTED. Reserves the interface for future BingX
    Futures API integration. No exchange connection exists. NEVER
    selected by get_provider_for_ticker()/get_default_provider() below -
    routing to an unimplemented placeholder would raise NotImplementedError
    for every single call, regardless of ticker."""

    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        raise NotImplementedError(
            "BingXFuturesProvider is not implemented yet. This class exists to "
            "reserve the interface for future BingX Futures API integration."
        )

    def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError(
            "BingXFuturesProvider is not implemented yet. This class exists to "
            "reserve the interface for future BingX Futures API integration."
        )


def _normalize_ticker(ticker: str) -> str:
    """
    Strips common separators and normalizes case so different formats of
    the same symbol compare equal - 'ETH/USDT', 'eth-usdt', 'ETH_USDT',
    and 'ETHUSDT' all normalize to 'ETHUSDT'. This is what the previous
    exact-match routing lacked, causing any non-canonical ticker format
    to silently fail its Binance-membership check and fall through to
    Yahoo (see module docstring).
    """
    return re.sub(r"[/\-_\s]", "", ticker).upper()


def _is_binance_ticker(ticker: str) -> bool:
    """
    True if `ticker`, in any common separator/case style, refers to a
    Binance Futures symbol. Checks symbols_config.BINANCE_SYMBOLS first
    (the explicit, verified registry) via a normalized comparison, then
    falls back to a naming-convention check (ends in USDT, not a Yahoo
    futures-style ticker) so a Binance-style ticker not yet added to that
    registry still routes correctly rather than silently defaulting to
    Yahoo.
    """
    normalized = _normalize_ticker(ticker)

    try:
        from symbols_config import BINANCE_SYMBOLS
        registered = {_normalize_ticker(s) for s in BINANCE_SYMBOLS}
        if normalized in registered:
            return True
    except ImportError:
        pass

    return normalized.endswith("USDT") and "=" not in ticker


def _is_yahoo_ticker(ticker: str) -> bool:
    """
    True if `ticker` matches Yahoo Finance's known formats used elsewhere
    in this project: futures contracts ('GC=F', 'SI=F') or hyphenated
    spot pairs ('BTC-USD', 'ETH-USD'). Deliberately checked only AFTER
    _is_binance_ticker() by the caller, so a ticker like 'BTC-USDT'
    (hyphenated Binance-style, if it ever occurred) would still correctly
    resolve to Binance via the USDT-suffix check first.
    """
    return "=F" in ticker or ("-" in ticker and not _normalize_ticker(ticker).endswith("USDT"))


def get_default_provider() -> MarketDataProvider:
    """
    Generic default for callers that don't have a specific ticker to
    route by. Returns YahooFinanceProvider() - restored to its original,
    correct behavior. (A prior edit had this hardcoded to
    BingXFuturesProvider(), an unimplemented placeholder that would raise
    NotImplementedError for every call - see module docstring.)
    paper_trade_monitor.py's real per-trade routing goes through
    get_provider_for_ticker() below, not this function.
    """
    return YahooFinanceProvider()


def get_provider_for_ticker(ticker: str) -> MarketDataProvider:
    """
    Routes a specific ticker to the correct provider based on its format,
    regardless of separator style or case (ETHUSDT, ETH/USDT, eth-usdt
    all route identically to Binance; BTC-USD, GC=F route to Yahoo).
    Never routes to BingXFuturesProvider, since it is an unimplemented
    placeholder - see module docstring for why this matters and what the
    previous bug was.
    """
    if _is_binance_ticker(ticker):
        return BinanceFuturesProvider()
    if _is_yahoo_ticker(ticker):
        return YahooFinanceProvider()

    # Ticker format doesn't match any known pattern. Rather than silently
    # guessing (the root cause of the original bug), default to Binance -
    # this project's live market-data target - while making the
    # ambiguity visible instead of failing deep inside a provider with a
    # confusing, unrelated-looking error.
    import warnings
    warnings.warn(
        f"get_provider_for_ticker: {ticker!r} did not match any known Binance or "
        f"Yahoo ticker format - defaulting to BinanceFuturesProvider. If this ticker "
        f"is a Yahoo-style symbol, add a case for it to _is_yahoo_ticker()."
    )
    return BinanceFuturesProvider()
