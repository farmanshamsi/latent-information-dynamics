from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.features.microstructure.engine import (
    MicrostructureFeatureEngine,
    MicrostructureFeatureError,
    as_feature_observations,
)
from latent_information_dynamics.market_data.book import (
    L2OrderBook,
)
from latent_information_dynamics.market_data.models import (
    AggressorSide,
    BookLevel,
    BookUpdate,
    L2Delta,
    L2Snapshot,
    SequenceRange,
    Side,
    Trade,
)


UTC = timezone.utc


def ts(
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(
        2026,
        1,
        1,
        12,
        minute,
        second,
        tzinfo=UTC,
    )


def make_book() -> L2OrderBook:
    snapshot = L2Snapshot(
        exchange="binance",
        symbol="BTC-USDT",
        event_time=ts(0, 0),
        receive_time=ts(0, 1),
        sequence=100,
        bids=(
            BookLevel(
                price=100.0,
                size=2.0,
            ),
            BookLevel(
                price=99.0,
                size=3.0,
            ),
        ),
        asks=(
            BookLevel(
                price=101.0,
                size=1.0,
            ),
            BookLevel(
                price=102.0,
                size=2.0,
            ),
        ),
    )

    book = L2OrderBook()

    book.initialize(
        snapshot
    )

    return book


def make_trade(
    *,
    trade_id: str,
    minute: int,
    size: float,
    side: AggressorSide,
) -> Trade:
    return Trade(
        exchange="binance",
        symbol="BTC-USDT",
        trade_id=trade_id,
        event_time=ts(
            minute,
            0,
        ),
        receive_time=ts(
            minute,
            1,
        ),
        price=100.0,
        size=size,
        aggressor_side=side,
    )


def test_initial_snapshot_has_zero_return_and_ofi() -> None:
    engine = MicrostructureFeatureEngine(
        depth_levels=2
    )

    state = engine.snapshot(
        make_book()
    )

    assert state.midpoint_log_return == 0.0
    assert state.order_flow_imbalance == 0.0


def test_relative_spread_is_in_basis_points() -> None:
    engine = MicrostructureFeatureEngine()

    state = engine.snapshot(
        make_book()
    )

    expected = (
        1.0
        / 100.5
        * 10_000.0
    )

    assert (
        state.relative_spread_bps
        == pytest.approx(
            expected
        )
    )


def test_microprice_displacement_is_positive_with_bid_pressure() -> None:
    engine = MicrostructureFeatureEngine()

    state = engine.snapshot(
        make_book()
    )

    assert (
        state.top_level_imbalance
        > 0
    )

    assert (
        state.microprice_displacement_bps
        > 0
    )


def test_depth_imbalance() -> None:
    engine = MicrostructureFeatureEngine(
        depth_levels=2
    )

    state = engine.snapshot(
        make_book()
    )

    expected = (
        5.0 - 3.0
    ) / (
        5.0 + 3.0
    )

    assert (
        state.depth_imbalance
        == pytest.approx(
            expected
        )
    )


def test_buy_trade_produces_positive_trade_imbalance() -> None:
    engine = MicrostructureFeatureEngine()

    engine.on_trade(
        make_trade(
            trade_id="1",
            minute=0,
            size=2.0,
            side=AggressorSide.BUY,
        )
    )

    state = engine.snapshot(
        make_book()
    )

    assert state.trade_count == 1
    assert state.trade_imbalance == 1.0
    assert state.notional_imbalance == 1.0


def test_balanced_buy_sell_flow() -> None:
    engine = MicrostructureFeatureEngine()

    engine.on_trade(
        make_trade(
            trade_id="1",
            minute=0,
            size=2.0,
            side=AggressorSide.BUY,
        )
    )

    engine.on_trade(
        make_trade(
            trade_id="2",
            minute=0,
            size=2.0,
            side=AggressorSide.SELL,
        )
    )

    state = engine.snapshot(
        make_book()
    )

    assert state.trade_count == 2

    assert (
        state.trade_imbalance
        == pytest.approx(
            0.0
        )
    )


def test_future_trade_is_not_used() -> None:
    engine = MicrostructureFeatureEngine()

    engine.on_trade(
        make_trade(
            trade_id="future",
            minute=5,
            size=100.0,
            side=AggressorSide.BUY,
        )
    )

    state = engine.snapshot(
        make_book()
    )

    assert state.trade_count == 0
    assert state.traded_size == 0.0
    assert state.trade_imbalance == 0.0


def test_old_trade_expires_from_window() -> None:
    engine = MicrostructureFeatureEngine(
        trade_window=timedelta(
            minutes=2
        )
    )

    engine.on_trade(
        make_trade(
            trade_id="old",
            minute=0,
            size=10.0,
            side=AggressorSide.BUY,
        )
    )

    book = make_book()

    book.apply(
        L2Delta(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(5, 0),
            receive_time=ts(5, 1),
            sequence=SequenceRange(
                first=101,
                last=101,
            ),
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=2.5,
                ),
            ),
        )
    )

    state = engine.snapshot(
        book
    )

    assert state.trade_count == 0


def test_book_change_generates_order_flow_imbalance() -> None:
    engine = MicrostructureFeatureEngine()

    book = make_book()

    engine.snapshot(
        book
    )

    book.apply(
        L2Delta(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(1, 0),
            receive_time=ts(1, 1),
            sequence=SequenceRange(
                first=101,
                last=101,
            ),
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=4.0,
                ),
            ),
        )
    )

    state = engine.snapshot(
        book
    )

    assert (
        state.order_flow_imbalance
        > 0
    )


def test_midpoint_return_changes_after_book_move() -> None:
    engine = MicrostructureFeatureEngine()

    book = make_book()

    first = engine.snapshot(
        book
    )

    book.apply(
        L2Delta(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(1, 0),
            receive_time=ts(1, 1),
            sequence=SequenceRange(
                first=101,
                last=101,
            ),
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.5,
                    size=2.0,
                ),
            ),
        )
    )

    second = engine.snapshot(
        book
    )

    assert (
        second.midpoint
        > first.midpoint
    )

    assert (
        second.midpoint_log_return
        > 0
    )


def test_trade_receive_time_cannot_move_backwards() -> None:
    engine = MicrostructureFeatureEngine()

    engine.on_trade(
        make_trade(
            trade_id="later",
            minute=2,
            size=1.0,
            side=AggressorSide.BUY,
        )
    )

    with pytest.raises(
        MicrostructureFeatureError
    ):
        engine.on_trade(
            make_trade(
                trade_id="earlier",
                minute=1,
                size=1.0,
                side=AggressorSide.BUY,
            )
        )


def test_feature_observations_preserve_availability_time() -> None:
    engine = MicrostructureFeatureEngine()

    book = make_book()

    state = engine.snapshot(
        book
    )

    observations = as_feature_observations(
        state,
        source="binance-live",
        entity="BTC-USDT",
    )

    assert len(
        observations
    ) > 0

    for observation in observations:
        assert (
            observation.available_time
            == state.available_time
        )

        assert (
            observation.source
            == "binance-live"
        )

        assert (
            observation.entity
            == "BTC-USDT"
        )


def test_feature_observation_contains_ofi() -> None:
    engine = MicrostructureFeatureEngine()

    observations = as_feature_observations(
        engine.snapshot(
            make_book()
        ),
        source="binance-live",
        entity="BTC-USDT",
    )

    values = {
        observation.name:
            observation.value
        for observation in observations
    }

    assert "order_flow_imbalance" in values
    assert "microprice_displacement_bps" in values
    assert "trade_imbalance" in values
