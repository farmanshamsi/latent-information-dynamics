from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.features.kline_trade_flow import (
    KlineTradeFlowError,
    KlineTradeFlowIndex,
    feature_names,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)


UTC = timezone.utc


def make_bar(
    open_time: datetime,
    *,
    price: float = 100.0,
    volume: float = 10.0,
    taker_buy: float = 6.0,
    trade_count: int = 100,
    irregular: bool = False,
) -> KlineBar:
    close_time = (
        open_time
        + timedelta(minutes=5)
        - timedelta(milliseconds=1)
    )

    if irregular:
        close_time -= timedelta(
            milliseconds=1
        )

    return KlineBar(
        symbol="BTCUSDT",
        interval="5m",
        open_time=open_time,
        close_time=close_time,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=volume,
        quote_volume=volume * price,
        trade_count=trade_count,
        taker_buy_base_volume=taker_buy,
        taker_buy_quote_volume=
            taker_buy * price,
    )


def test_default_contract_has_20_features() -> None:
    names = feature_names()

    assert len(names) == 20
    assert len(set(names)) == 20

    assert (
        names[0]
        == "total_quantity_5m"
    )

    assert (
        names[-1]
        == "trade_count_4h"
    )


def test_quantity_flow_formula() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    bars = [
        make_bar(
            start,
            volume=10.0,
            taker_buy=7.0,
            trade_count=100,
        ),
        make_bar(
            start + timedelta(minutes=5),
            volume=20.0,
            taker_buy=8.0,
            trade_count=200,
        ),
        make_bar(
            start + timedelta(minutes=10),
            volume=30.0,
            taker_buy=20.0,
            trade_count=300,
        ),
    ]

    index = KlineTradeFlowIndex(
        bars
    )

    summary = index.summarize(
        decision_time=
            start
            + timedelta(minutes=15),
        label="15m",
        window=timedelta(minutes=15),
    )

    assert summary.total_quantity == 60.0

    # Signed:
    # (14 - 10) + (16 - 20) + (40 - 30)
    # = 4 - 4 + 10 = 10.
    assert summary.signed_quantity == 10.0

    assert summary.quantity_imbalance == (
        10.0 / 60.0
    )

    assert summary.trade_count == 600


def test_features_are_invariant_to_prices() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    first = []
    second = []

    for i in range(3):
        timestamp = (
            start
            + timedelta(
                minutes=5 * i
            )
        )

        first.append(
            make_bar(
                timestamp,
                price=
                    30000.0
                    + i,
                volume=
                    10.0 + i,
                taker_buy=
                    6.0 + i,
                trade_count=
                    100 + i,
            )
        )

        second.append(
            make_bar(
                timestamp,
                price=
                    90000.0
                    + i * 1000,
                volume=
                    10.0 + i,
                taker_buy=
                    6.0 + i,
                trade_count=
                    100 + i,
            )
        )

    first_summary = (
        KlineTradeFlowIndex(
            first
        ).summarize(
            decision_time=
                start
                + timedelta(minutes=15),
            label="15m",
            window=timedelta(minutes=15),
        )
    )

    second_summary = (
        KlineTradeFlowIndex(
            second
        ).summarize(
            decision_time=
                start
                + timedelta(minutes=15),
            label="15m",
            window=timedelta(minutes=15),
        )
    )

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
        first_summary.trade_count
        == second_summary.trade_count
    )


def test_rejects_missing_bar_in_window() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    bars = [
        make_bar(start),
        make_bar(
            start
            + timedelta(minutes=10)
        ),
    ]

    index = KlineTradeFlowIndex(
        bars
    )

    with pytest.raises(
        KlineTradeFlowError
    ):
        index.summarize(
            decision_time=
                start
                + timedelta(minutes=15),
            label="15m",
            window=
                timedelta(minutes=15),
        )


def test_rejects_irregular_bar_in_window() -> None:
    start = datetime(
        2024, 1, 1,
        tzinfo=UTC,
    )

    bars = [
        make_bar(start),
        make_bar(
            start
            + timedelta(minutes=5),
            irregular=True,
        ),
        make_bar(
            start
            + timedelta(minutes=10)
        ),
    ]

    index = KlineTradeFlowIndex(
        bars
    )

    with pytest.raises(
        KlineTradeFlowError
    ):
        index.summarize(
            decision_time=
                start
                + timedelta(minutes=15),
            label="15m",
            window=
                timedelta(minutes=15),
        )
