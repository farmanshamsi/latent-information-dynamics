from datetime import datetime, timezone

import csv
import pytest

from latent_information_dynamics.market_data.binance_historical import (
    BinanceSpotKlineClient,
    HistoricalMarketDataError,
    KlineBar,
    parse_binance_kline,
    write_klines_csv,
)


UTC = timezone.utc


def ms(hour: int, minute: int) -> int:
    return int(
        datetime(
            2026, 1, 1,
            hour, minute,
            tzinfo=UTC,
        ).timestamp() * 1000
    )


def sample_row(open_time: int | None = None) -> list:
    if open_time is None:
        open_time = ms(0, 0)

    return [
        open_time,
        "100.0",
        "105.0",
        "99.0",
        "102.0",
        "10.0",
        open_time + 299999,
        "1010.0",
        25,
        "6.0",
        "606.0",
        "0",
    ]


def test_parse_binance_kline() -> None:
    bar = parse_binance_kline(
        sample_row(),
        symbol="BTCUSDT",
        interval="5m",
    )

    assert bar.symbol == "BTCUSDT"
    assert bar.interval == "5m"
    assert bar.open == 100.0
    assert bar.close == 102.0
    assert bar.trade_count == 25


def test_bar_availability_is_after_close() -> None:
    bar = parse_binance_kline(
        sample_row(),
        symbol="BTCUSDT",
        interval="5m",
    )

    assert bar.available_time > bar.close_time


def test_taker_buy_fraction() -> None:
    bar = parse_binance_kline(
        sample_row(),
        symbol="BTCUSDT",
        interval="5m",
    )

    assert bar.taker_buy_fraction == pytest.approx(0.6)


def test_invalid_ohlc_is_rejected() -> None:
    with pytest.raises(HistoricalMarketDataError):
        KlineBar(
            symbol="BTCUSDT",
            interval="5m",
            open_time=datetime(
                2026, 1, 1,
                tzinfo=UTC,
            ),
            close_time=datetime(
                2026, 1, 1,
                0, 5,
                tzinfo=UTC,
            ),
            open=100.0,
            high=101.0,
            low=99.0,
            close=105.0,
            volume=1.0,
            quote_volume=100.0,
            trade_count=1,
            taker_buy_base_volume=0.5,
            taker_buy_quote_volume=50.0,
        )


class FakeClient(BinanceSpotKlineClient):
    def __init__(self, pages) -> None:
        super().__init__()
        self.pages = list(pages)

    def _request_json(self, path, parameters):
        if not self.pages:
            return []

        return self.pages.pop(0)


def test_fetch_range_paginates() -> None:
    first = [
        sample_row(ms(0, 0)),
        sample_row(ms(0, 5)),
    ]

    second = [
        sample_row(ms(0, 10)),
    ]

    client = FakeClient([first, second])

    bars = client.fetch_range(
        symbol="BTCUSDT",
        interval="5m",
        start_time=datetime(
            2026, 1, 1,
            0, 0,
            tzinfo=UTC,
        ),
        end_time=datetime(
            2026, 1, 1,
            1, 0,
            tzinfo=UTC,
        ),
        limit=2,
    )

    assert len(bars) == 3

    assert [
        bar.open_time.minute
        for bar in bars
    ] == [0, 5, 10]


def test_csv_is_sorted(tmp_path) -> None:
    later = parse_binance_kline(
        sample_row(ms(0, 5)),
        symbol="BTCUSDT",
        interval="5m",
    )

    earlier = parse_binance_kline(
        sample_row(ms(0, 0)),
        symbol="BTCUSDT",
        interval="5m",
    )

    output = write_klines_csv(
        (later, earlier),
        tmp_path / "bars.csv",
    )

    with output.open(
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["open_time"] < rows[1]["open_time"]
