from __future__ import annotations

import math

from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.features.price.baseline import (
    PriceFeatureError,
    build_price_feature_snapshot,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)


UTC = timezone.utc
FIVE_MINUTES = timedelta(minutes=5)


def make_bar(
    open_time: datetime,
    price: float,
    *,
    irregular: bool = False,
) -> KlineBar:
    if irregular:
        close_time = (
            open_time
            + timedelta(minutes=2)
        )
    else:
        close_time = (
            open_time
            + FIVE_MINUTES
            - timedelta(milliseconds=1)
        )

    return KlineBar(
        symbol="BTCUSDT",
        interval="5m",
        open_time=open_time,
        close_time=close_time,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=10.0,
        quote_volume=price * 10.0,
        trade_count=100,
        taker_buy_base_volume=5.0,
        taker_buy_quote_volume=price * 5.0,
    )


def make_history(
    decision_time: datetime,
) -> list[KlineBar]:
    start = (
        decision_time
        - timedelta(hours=24)
        - FIVE_MINUTES
    )

    bars: list[KlineBar] = []

    cursor = start
    index = 0

    while cursor < decision_time:
        bars.append(
            make_bar(
                cursor,
                100.0 + index * 0.1,
            )
        )

        cursor += FIVE_MINUTES
        index += 1

    return bars


def test_builds_expected_feature_set() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    snapshot = build_price_feature_snapshot(
        make_history(decision),
        decision,
    )

    assert set(snapshot.features) == {
        "return_5m",
        "return_30m",
        "return_1h",
        "return_4h",
        "return_12h",
        "return_24h",
        "rv_1h",
        "rv_4h",
        "rv_24h",
        "range_4h",
        "range_24h",
    }

    assert all(
        math.isfinite(value)
        for value in snapshot.features.values()
    )


def test_five_minute_return_is_correct() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    bars = make_history(decision)

    snapshot = build_price_feature_snapshot(
        bars,
        decision,
    )

    expected = math.log(
        bars[-1].close
        / bars[-2].close
    )

    assert snapshot.features["return_5m"] == pytest.approx(
        expected
    )


def test_future_bar_cannot_change_features() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    bars = make_history(decision)

    original = build_price_feature_snapshot(
        bars,
        decision,
    )

    future_bar = make_bar(
        decision,
        999999.0,
    )

    with_future = build_price_feature_snapshot(
        [*bars, future_bar],
        decision,
    )

    assert with_future.features == original.features


def test_missing_historical_bar_is_rejected() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    missing_time = (
        decision
        - timedelta(minutes=15)
    )

    bars = [
        bar
        for bar in make_history(decision)
        if bar.open_time != missing_time
    ]

    with pytest.raises(
        PriceFeatureError,
        match="Missing bar",
    ):
        build_price_feature_snapshot(
            bars,
            decision,
        )


def test_irregular_source_bar_is_rejected() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    bars = make_history(decision)

    irregular_time = (
        decision
        - timedelta(minutes=10)
    )

    bars = [
        (
            make_bar(
                bar.open_time,
                bar.close,
                irregular=True,
            )
            if bar.open_time == irregular_time
            else bar
        )
        for bar in bars
    ]

    with pytest.raises(
        PriceFeatureError,
        match="Irregular bar",
    ):
        build_price_feature_snapshot(
            bars,
            decision,
        )


def test_decision_time_must_be_four_hour_boundary() -> None:
    bars = make_history(
        datetime(
            2025, 1, 2, 4, 0,
            tzinfo=UTC,
        )
    )

    with pytest.raises(
        PriceFeatureError,
        match="4-hour boundary",
    ):
        build_price_feature_snapshot(
            bars,
            datetime(
                2025, 1, 2, 2, 0,
                tzinfo=UTC,
            ),
        )


def test_naive_decision_time_is_rejected() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    bars = make_history(decision)

    with pytest.raises(
        PriceFeatureError,
        match="timezone-aware",
    ):
        build_price_feature_snapshot(
            bars,
            datetime(
                2025, 1, 2, 4, 0,
            ),
        )
