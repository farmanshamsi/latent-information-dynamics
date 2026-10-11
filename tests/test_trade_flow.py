from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.features.trade_flow import (
    MultiHorizonTradeFlowEngine,
    TradeFlowError,
)
from latent_information_dynamics.market_data.models import (
    AggressorSide,
    Trade,
)


UTC = timezone.utc


def make_trade(
    identifier: int,
    time: datetime,
    *,
    price: float,
    size: float,
    side: AggressorSide,
) -> Trade:
    return Trade(
        exchange="binance-spot",
        symbol="BTCUSDT",
        trade_id=str(identifier),
        event_time=time,
        receive_time=time,
        price=price,
        size=size,
        aggressor_side=side,
        sequence=identifier,
    )


def test_quantity_imbalance() -> None:
    start = datetime(
        2024, 1, 1, 0, 0,
        tzinfo=UTC,
    )

    engine = MultiHorizonTradeFlowEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.on_trade(
        make_trade(
            1,
            start + timedelta(minutes=1),
            price=40000.0,
            size=3.0,
            side=AggressorSide.BUY,
        )
    )

    engine.on_trade(
        make_trade(
            2,
            start + timedelta(minutes=2),
            price=50000.0,
            size=1.0,
            side=AggressorSide.SELL,
        )
    )

    summary = engine.summaries(
        as_of=
            start + timedelta(minutes=5)
    )["5m"]

    assert summary.total_quantity == 4.0
    assert summary.signed_quantity == 2.0
    assert summary.quantity_imbalance == 0.5
    assert summary.count_imbalance == 0.0


def test_features_do_not_depend_on_trade_price() -> None:
    start = datetime(
        2024, 1, 1, 0, 0,
        tzinfo=UTC,
    )

    first = MultiHorizonTradeFlowEngine(
        windows={
            "1h": timedelta(hours=1)
        }
    )

    second = MultiHorizonTradeFlowEngine(
        windows={
            "1h": timedelta(hours=1)
        }
    )

    specifications = (
        (
            1,
            2.0,
            AggressorSide.BUY,
        ),
        (
            2,
            1.5,
            AggressorSide.SELL,
        ),
        (
            3,
            4.0,
            AggressorSide.BUY,
        ),
    )

    for identifier, size, side in specifications:
        event_time = (
            start
            + timedelta(
                minutes=identifier * 10
            )
        )

        first.on_trade(
            make_trade(
                identifier,
                event_time,
                price=
                    30000.0
                    + identifier,
                size=size,
                side=side,
            )
        )

        second.on_trade(
            make_trade(
                identifier,
                event_time,
                price=
                    90000.0
                    + identifier * 1000,
                size=size,
                side=side,
            )
        )

    first_summary = first.summaries(
        as_of=
            start + timedelta(hours=1)
    )["1h"]

    second_summary = second.summaries(
        as_of=
            start + timedelta(hours=1)
    )["1h"]

    assert (
        first_summary.total_quantity
        == second_summary.total_quantity
    )

    assert (
        first_summary.signed_quantity
        == second_summary.signed_quantity
    )

    assert (
        first_summary.quantity_imbalance
        == second_summary.quantity_imbalance
    )

    assert (
        first_summary.count_imbalance
        == second_summary.count_imbalance
    )


def test_old_trades_are_removed() -> None:
    start = datetime(
        2024, 1, 1, 0, 0,
        tzinfo=UTC,
    )

    engine = MultiHorizonTradeFlowEngine(
        windows={
            "1m": timedelta(minutes=1)
        }
    )

    engine.on_trade(
        make_trade(
            1,
            start,
            price=40000.0,
            size=10.0,
            side=AggressorSide.BUY,
        )
    )

    summary = engine.summaries(
        as_of=
            start
            + timedelta(
                minutes=1,
                seconds=1,
            )
    )["1m"]

    assert summary.aggregate_trade_count == 0
    assert summary.total_quantity == 0.0


def test_rejects_unknown_aggressor() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    engine = MultiHorizonTradeFlowEngine()

    with pytest.raises(
        TradeFlowError
    ):
        engine.on_trade(
            make_trade(
                1,
                start,
                price=40000.0,
                size=1.0,
                side=AggressorSide.UNKNOWN,
            )
        )


def test_rejects_backwards_trade_time() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    engine = MultiHorizonTradeFlowEngine()

    engine.on_trade(
        make_trade(
            2,
            start + timedelta(seconds=2),
            price=40000.0,
            size=1.0,
            side=AggressorSide.BUY,
        )
    )

    with pytest.raises(
        TradeFlowError
    ):
        engine.on_trade(
            make_trade(
                1,
                start + timedelta(seconds=1),
                price=40000.0,
                size=1.0,
                side=AggressorSide.BUY,
            )
        )
