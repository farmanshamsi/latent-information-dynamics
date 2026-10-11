from datetime import datetime, timedelta, timezone

from latent_information_dynamics.features.price.baseline import (
    build_price_feature_snapshot,
)
from latent_information_dynamics.features.price.engine import (
    PriceBaselineEngine,
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
        price = 100.0 + index * 0.1

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


def test_engine_matches_reference_function() -> None:
    decision = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    bars = make_bars(
        decision
        - timedelta(hours=24)
        - STEP,
        decision,
    )

    expected = build_price_feature_snapshot(
        bars,
        decision,
    )

    engine = PriceBaselineEngine(bars)

    actual = engine.snapshot(decision)

    assert actual == expected


def test_engine_reuses_history_for_multiple_decisions() -> None:
    first = datetime(
        2025, 1, 2, 4, 0,
        tzinfo=UTC,
    )

    second = first + timedelta(hours=4)

    bars = make_bars(
        first
        - timedelta(hours=24)
        - STEP,
        second,
    )

    engine = PriceBaselineEngine(bars)

    first_snapshot = engine.snapshot(first)
    second_snapshot = engine.snapshot(second)

    assert engine.bar_count == len(bars)
    assert first_snapshot.decision_time == first
    assert second_snapshot.decision_time == second
    assert first_snapshot.features != second_snapshot.features
