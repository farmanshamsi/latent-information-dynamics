from datetime import datetime, timedelta, timezone

from latent_information_dynamics.datasets.target_materialization import (
    materialize_binary_targets,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)
from latent_information_dynamics.targets.binary import (
    TieRule,
)
from latent_information_dynamics.targets.binance_reference import (
    build_kline_index,
    reference_at_boundary,
)


UTC = timezone.utc
STEP = timedelta(minutes=5)


def make_bar(
    open_time: datetime,
    close: float,
) -> KlineBar:
    return KlineBar(
        symbol="BTCUSDT",
        interval="5m",
        open_time=open_time,
        close_time=(
            open_time
            + STEP
            - timedelta(milliseconds=1)
        ),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10.0,
        quote_volume=close * 10.0,
        trade_count=100,
        taker_buy_base_volume=5.0,
        taker_buy_quote_volume=close * 5.0,
    )


def test_boundary_reference_uses_previous_completed_bar() -> None:
    boundary = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    bar = make_bar(
        boundary - STEP,
        101.5,
    )

    observation = reference_at_boundary(
        build_kline_index([bar]),
        boundary=boundary,
    )

    assert observation.reference_time == boundary
    assert observation.available_time == boundary
    assert observation.price == 101.5


def test_up_target() -> None:
    decision = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    resolution = (
        decision
        + timedelta(hours=4)
    )

    bars = [
        make_bar(
            decision - STEP,
            100.0,
        ),
        make_bar(
            resolution - STEP,
            105.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[decision],
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.targets[0].outcome == 1


def test_down_target() -> None:
    decision = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    resolution = (
        decision
        + timedelta(hours=4)
    )

    bars = [
        make_bar(
            decision - STEP,
            105.0,
        ),
        make_bar(
            resolution - STEP,
            100.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[decision],
    )

    assert result.targets[0].outcome == 0


def test_tie_rule_up_matches_primary_target_definition() -> None:
    decision = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    resolution = (
        decision
        + timedelta(hours=4)
    )

    bars = [
        make_bar(
            decision - STEP,
            100.0,
        ),
        make_bar(
            resolution - STEP,
            100.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[decision],
        tie_rule=TieRule.UP,
    )

    assert result.targets[0].outcome == 1
    assert result.targets[0].arithmetic_return == 0.0


def test_missing_end_reference_rejects_target() -> None:
    decision = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    bars = [
        make_bar(
            decision - STEP,
            100.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[decision],
    )

    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert (
        "Missing reference bar"
        in result.rejected[0].reason
    )


def test_target_returns_are_correct() -> None:
    decision = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    resolution = (
        decision
        + timedelta(hours=4)
    )

    bars = [
        make_bar(
            decision - STEP,
            100.0,
        ),
        make_bar(
            resolution - STEP,
            110.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[decision],
    )

    target = result.targets[0]

    assert abs(
        target.arithmetic_return
        - 0.10
    ) < 1e-12

    assert target.log_return > 0.0


def test_multiple_decisions_are_materialized_independently() -> None:
    first = datetime(
        2025, 1, 1, 4, 0,
        tzinfo=UTC,
    )

    second = first + timedelta(
        hours=4
    )

    third = second + timedelta(
        hours=4
    )

    bars = [
        make_bar(
            first - STEP,
            100.0,
        ),
        make_bar(
            second - STEP,
            101.0,
        ),
        make_bar(
            third - STEP,
            99.0,
        ),
    ]

    result = materialize_binary_targets(
        bars,
        decision_times=[
            first,
            second,
        ],
    )

    assert result.requested_decisions == 2
    assert result.accepted_count == 2
    assert [
        target.outcome
        for target in result.targets
    ] == [1, 0]
