"""
Market Data Abstraction Layer.

Provides a common interface for OHLCV/current-price data. Binance Futures is
used when reachable; if Binance returns a geographic/API access error, a
completely free Yahoo Finance chart endpoint is used automatically as a
read-only fallback. This is intended for paper-trade monitoring only.
"""

import re
from abc import ABC, abstractmethod
import pandas as pd

from symbols_config import is_tradfi_perpetual


class MarketDataProvider(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance provider used by the project for non-Binance tickers."""

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
        df = self.get_bars(symbol, interval="1m", period="1d")
        if df.empty:
            raise ValueError(f"YahooFinanceProvider: no recent bars available for {symbol}")
        return float(df["Close"].iloc[-1])


class YahooChartFallbackProvider(MarketDataProvider):
    """Free, keyless Yahoo chart API fallback for Binance-blocked runners.

    This deliberately uses the public chart endpoint directly, so no API key,
    paid service, proxy, VPN, or additional package is required.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    SYMBOL_MAP = {
        "XAUUSDT": "GC=F",   # Gold futures proxy
        "XAGUSDT": "SI=F",   # Silver futures proxy
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "XRPUSDT": "XRP-USD",
        "BNBUSDT": "BNB-USD",
    }

    INTERVAL_MAP = {
        "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
        "30m": "30m", "60m": "60m", "1h": "60m", "90m": "90m",
        "1d": "1d", "1wk": "1wk", "1mo": "1mo",
    }

    def _symbol(self, symbol: str) -> str:
        normalized = _normalize_ticker(symbol)
        return self.SYMBOL_MAP.get(normalized, normalized)

    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        import requests

        yahoo_symbol = self._symbol(symbol)
        yahoo_interval = self.INTERVAL_MAP.get(interval, interval)
        params = {"interval": yahoo_interval, "events": "history", "includePrePost": "true"}

        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            params["period1"] = int(start_ts.timestamp())
            params["period2"] = int(pd.Timestamp.now(tz="UTC").timestamp())
        else:
            # Keep the default lookback small enough for Yahoo's intraday rules.
            lookback = _parse_period_to_timedelta(period or "5d")
            params["period1"] = int((pd.Timestamp.now(tz="UTC") - lookback).timestamp())
            params["period2"] = int(pd.Timestamp.now(tz="UTC").timestamp())

        resp = requests.get(
            f"{self.BASE_URL}/{yahoo_symbol}",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        result = (payload.get("chart") or {}).get("result")
        if not result:
            raise ValueError(f"Yahoo fallback returned no chart data for {symbol} ({yahoo_symbol})")

        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        rows = []
        for i, ts in enumerate(timestamps):
            o = (quote.get("open") or [None])[i]
            h = (quote.get("high") or [None])[i]
            l = (quote.get("low") or [None])[i]
            c = (quote.get("close") or [None])[i]
            if None in (o, h, l, c):
                continue
            rows.append({
                "Open": float(o), "High": float(h),
                "Low": float(l), "Close": float(c),
            })

        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

        index = pd.to_datetime([timestamps[i] for i, ts in enumerate(timestamps)
                                if all((quote.get(k) or [None])[i] is not None
                                       for k in ("open", "high", "low", "close"))],
                               unit="s", utc=True)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="open_time"))

    def get_current_price(self, symbol: str) -> float:
        df = self.get_bars(symbol, interval="1m", period="1d")
        if df.empty:
            raise ValueError(f"Yahoo fallback: no current price available for {symbol}")
        return float(df["Close"].iloc[-1])


def _parse_period_to_timedelta(period: str) -> pd.Timedelta:
    match = re.match(r"^(\d+)([dhm])$", str(period).strip())
    if not match:
        raise ValueError(f"Unsupported period format: {period!r}")
    n, unit = int(match.group(1)), match.group(2)
    return pd.Timedelta(**{"d": "days", "h": "hours", "m": "minutes"}[unit])


class BinanceFuturesProvider(MarketDataProvider):
    """Binance public Futures provider with automatic free Yahoo fallback."""

    BASE_URL = "https://fapi.binance.com"
    KLINES_LIMIT = 1500
    MAX_PAGES = 20

    def _fallback(self, symbol: str, operation: str, error: Exception):
        print(
            f"Binance market data unavailable for {symbol} during {operation} "
            f"({error}); switching to free Yahoo fallback."
        )
        return YahooChartFallbackProvider()

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
            try:
                return self._fetch_paginated(
                    f"{self.BASE_URL}/fapi/v1/klines",
                    {"symbol": symbol, "interval": interval}, start_ts,
                )
            except requests.RequestException:
                if is_tradfi_perpetual(symbol):
                    return self._fetch_paginated(
                        f"{self.BASE_URL}/fapi/v1/continuousKlines",
                        {"pair": symbol, "contractType": "TRADIFI_PERPETUAL", "interval": interval}, start_ts,
                    )
                raise
        except Exception as exc:
            return self._fallback(symbol, "OHLC", exc).get_bars(symbol, interval, start=start, period=period)

    def _fetch_paginated(self, url: str, base_params: dict, start_ts) -> pd.DataFrame:
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
            if last_open_ms >= now_ms or len(page) < self.KLINES_LIMIT:
                break
            current_start = pd.Timestamp(last_open_ms + 1, unit="ms", tz="UTC")

        return _parse_klines(all_rows)

    def get_current_price(self, symbol: str) -> float:
        import requests
        symbol = _normalize_ticker(symbol)
        try:
            resp = requests.get(
                f"{self.BASE_URL}/fapi/v1/ticker/price",
                params={"symbol": symbol},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if "price" not in data:
                raise ValueError(f"Unexpected Binance ticker response for {symbol}: {data}")
            return float(data["price"])
        except Exception as exc:
            return self._fallback(symbol, "current price", exc).get_current_price(symbol)


def _parse_klines(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    times, rows = [], []
    for k in raw:
        times.append(pd.Timestamp(int(k[0]), unit="ms", tz="UTC"))
        rows.append({"Open": float(k[1]), "High": float(k[2]), "Low": float(k[3]), "Close": float(k[4])})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name="open_time"))


class BingXFuturesProvider(MarketDataProvider):
    """Reserved placeholder; never selected automatically."""

    def get_bars(self, symbol: str, interval: str, start=None, period=None) -> pd.DataFrame:
        raise NotImplementedError("BingXFuturesProvider is not implemented yet.")

    def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError("BingXFuturesProvider is not implemented yet.")


def _normalize_ticker(ticker: str) -> str:
    return re.sub(r"[/\-_\s]", "", str(ticker)).upper()


def _is_binance_ticker(ticker: str) -> bool:
    normalized = _normalize_ticker(ticker)
    try:
        from symbols_config import BINANCE_SYMBOLS
        if normalized in {_normalize_ticker(s) for s in BINANCE_SYMBOLS}:
            return True
    except ImportError:
        pass
    return normalized.endswith("USDT") and "=" not in str(ticker)


def _is_yahoo_ticker(ticker: str) -> bool:
    return "=F" in str(ticker) or ("-" in str(ticker) and not _normalize_ticker(ticker).endswith("USDT"))


def get_default_provider() -> MarketDataProvider:
    return YahooFinanceProvider()


def get_provider_for_ticker(ticker: str) -> MarketDataProvider:
    if _is_binance_ticker(ticker):
        return BinanceFuturesProvider()
    if _is_yahoo_ticker(ticker):
        return YahooFinanceProvider()
    import warnings
    warnings.warn(
        f"get_provider_for_ticker: {ticker!r} did not match known formats; "
        "defaulting to BinanceFuturesProvider."
    )
    return BinanceFuturesProvider()
