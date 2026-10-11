import pandas as pd

from paper_trade_monitor import resolve_open_trade


class FakeProvider:
    def __init__(self, bars):
        self.bars = bars
        self.requested = None

    def get_bars(self, symbol, interval, start=None, period=None):
        self.requested = (symbol, interval, start)
        return self.bars.copy()

    def get_current_price(self, symbol):
        return 100.0


def trade_row():
    return {
        "trade_id": "BTCUSDT-20200101T000000-BUY",
        "ticker": "BTC/USDT",
        "direction": "BUY",
        "entry": "100",
        "sl": "99",
        "tp": "103",
        "confidence": "HIGH",
        "entry_time_utc": "2020-01-01T00:00:00+00:00",
    }


def bars(rows, times):
    return pd.DataFrame(rows, index=pd.to_datetime(times, utc=True))


def test_signal_candle_is_excluded_from_exit_replay():
    # Signal candle has a TP-crossing high, but that movement happened before
    # the simulated entry at the signal candle close and must not close trade.
    frame = bars(
        [
            {"Open": 100, "High": 104, "Low": 98, "Close": 100},
            {"Open": 100, "High": 101, "Low": 99.5, "Close": 100.5},
        ],
        ["2020-01-01T00:00:00Z", "2020-01-01T00:15:00Z"],
    )
    provider = FakeProvider(frame)

    result = resolve_open_trade(trade_row(), provider)

    assert provider.requested[1] == "15m"
    assert provider.requested[2] == pd.Timestamp("2020-01-01T00:15:00Z")
    assert result["status"] == "STILL_OPEN"


def test_completed_post_entry_candle_can_trigger_take_profit():
    frame = bars(
        [
            {"Open": 100, "High": 101, "Low": 99.5, "Close": 100.5},
            {"Open": 100.5, "High": 103.5, "Low": 100, "Close": 103},
        ],
        ["2020-01-01T00:15:00Z", "2020-01-01T00:30:00Z"],
    )
    result = resolve_open_trade(trade_row(), FakeProvider(frame))

    assert result["status"] == "CLOSED"
    assert result["position"].exit_reason == "TAKE_PROFIT"
    assert result["position"].exit_price == 103
    assert result["duration_bars"] == 2
    assert result["duration_wall_clock"] == pd.Timedelta(minutes=30)


def test_incomplete_latest_candle_is_not_used():
    # The latest bar opens at the current 15-minute boundary and is unfinished.
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    frame = bars(
        [
            {"Open": 100, "High": 101, "Low": 99.5, "Close": 100.5},
            {"Open": 100.5, "High": 104, "Low": 100, "Close": 103.5},
        ],
        [now - pd.Timedelta(minutes=30), now],
    )
    result = resolve_open_trade(trade_row(), FakeProvider(frame))

    assert result["status"] == "STILL_OPEN"
    assert result.get("reason") == "no completed post-entry bars available this run"
