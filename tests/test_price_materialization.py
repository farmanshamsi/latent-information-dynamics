from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.datasets.price_materialization import (
    PriceMaterializationError,
    four_hour_boundaries,
    materialize_price_features,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)


UTC = timezone.utc
STEP = timedelta(minutes=5)


def make_bars(
    start: datetime,
    end: datetime,
) -> list[KlineBar]:
    bars = []
    cursor = start
    index = 0

    while cursor < end:
        price = 100.0 + index * 0.01

        bars.append(
            KlineBar(
                symbol="BTCUSDT",
                interval="5m",
                open_time=cursor,
                close_time=(
                    cursor
                    + STEP
                    - timedelta(milliseconds=1)
                ),
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
        )

        cursor += STEP
        index += 1

    return bars


def test_four_hour_boundaries() -> None:
    start = datetime(
        2025, 1, 1, 0, 0,
        tzinfo=UTC,
    )

    end = datetime(
        2025, 1, 2, 0, 0,
        tzinfo=UTC,
    )

    values = four_hour_boundaries(
        start,
        end,
    )

    assert len(values) == 6
    assert [value.hour for value in values] == [
        0, 4, 8, 12, 16, 20
    ]


def test_naive_start_is_rejected() -> None:
    with pytest.raises(
        PriceMaterializationError,
        match="timezone-aware",
    ):
        four_hour_boundaries(
            datetime(2025, 1, 1),
            datetime(
                2025, 1, 2,
                tzinfo=UTC,
            ),
        )


def test_materialization_accepts_complete_history() -> None:
    start = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    end = start + timedelta(hours=8)

    bars = make_bars(
        start
        - timedelta(hours=24)
        - STEP,
        end,
    )

    result = materialize_price_features(
        bars,
        start=start,
        end=end,
    )

    assert result.theoretical_decisions == 2
    assert result.accepted_count == 2
    assert result.rejected_count == 0


def test_materialization_records_gap_rejection() -> None:
    start = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    end = start + timedelta(hours=4)

    missing_time = (
        start
        - timedelta(minutes=15)
    )

    bars = [
        bar
        for bar in make_bars(
            start
            - timedelta(hours=24)
            - STEP,
            end,
        )
        if bar.open_time != missing_time
    ]

    result = materialize_price_features(
        bars,
        start=start,
        end=end,
    )

    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert "Missing bar" in result.rejected[0].reason


def test_rejection_does_not_stop_later_decisions() -> None:
    start = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    end = start + timedelta(hours=8)

    missing_time = (
        start
        - timedelta(minutes=15)
    )

    bars = [
        bar
        for bar in make_bars(
            start
            - timedelta(hours=24)
            - STEP,
            end,
        )
        if bar.open_time != missing_time
    ]

    result = materialize_price_features(
        bars,
        start=start,
        end=end,
    )

    assert result.theoretical_decisions == 2
    assert (
        result.accepted_count
        + result.rejected_count
        == 2
    )
