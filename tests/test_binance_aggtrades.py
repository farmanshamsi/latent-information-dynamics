from datetime import timezone

import pytest

from latent_information_dynamics.market_data.binance_aggtrades import (
    BinanceAggTradeError,
    parse_binance_aggtrade_row,
)
from latent_information_dynamics.market_data.models import (
    AggressorSide,
)


UTC = timezone.utc


def test_parse_buy_aggressor_millisecond_row() -> None:
    trade = parse_binance_aggtrade_row(
        [
            "100",
            "42000.50",
            "0.25",
            "200",
            "201",
            "1609459200123",
            "false",
            "true",
        ]
    )

    assert trade.trade_id == "100"
    assert trade.size == 0.25
    assert (
        trade.aggressor_side
        is AggressorSide.BUY
    )
    assert (
        trade.event_time.tzinfo
        == UTC
    )
    assert (
        trade.receive_time
        == trade.event_time
    )


def test_buyer_maker_means_sell_aggressor() -> None:
    trade = parse_binance_aggtrade_row(
        [
            "101",
            "42000.50",
            "0.75",
            "202",
            "204",
            "1609459201123",
            "true",
            "true",
        ]
    )

    assert (
        trade.aggressor_side
        is AggressorSide.SELL
    )


def test_parse_2025_microsecond_timestamp() -> None:
    trade = parse_binance_aggtrade_row(
        [
            "102",
            "95000.00",
            "0.10",
            "205",
            "205",
            "1735689600010866",
            "false",
            "true",
        ]
    )

    assert trade.event_time.year == 2025
    assert trade.event_time.month == 1
    assert trade.event_time.day == 1


def test_rejects_wrong_column_count() -> None:
    with pytest.raises(
        BinanceAggTradeError
    ):
        parse_binance_aggtrade_row(
            ["1", "2"]
        )


def test_rejects_invalid_boolean() -> None:
    with pytest.raises(
        BinanceAggTradeError
    ):
        parse_binance_aggtrade_row(
            [
                "100",
                "42000",
                "1",
                "1",
                "1",
                "1609459200000",
                "maybe",
                "true",
            ]
        )
