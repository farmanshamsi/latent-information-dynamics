from datetime import datetime, timezone

import pytest

from latent_information_dynamics.market_data.models import (
    AggressorSide,
    BookLevel,
    BookUpdate,
    L2Delta,
    L2Snapshot,
    MarketDataValidationError,
    SequenceIntegrityError,
    SequenceRange,
    SequenceTracker,
    Side,
    Trade,
)


UTC = timezone.utc


def ts(
    second: int = 0,
) -> datetime:
    return datetime(
        2026,
        1,
        1,
        12,
        0,
        second,
        tzinfo=UTC,
    )


def snapshot(
    *,
    sequence: int = 100,
) -> L2Snapshot:
    return L2Snapshot(
        exchange="binance",
        symbol="BTC-USDT",
        event_time=ts(0),
        receive_time=ts(1),
        sequence=sequence,
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
                size=1.5,
            ),
            BookLevel(
                price=102.0,
                size=4.0,
            ),
        ),
    )


def delta(
    *,
    first: int,
    last: int | None = None,
) -> L2Delta:
    return L2Delta(
        exchange="binance",
        symbol="BTC-USDT",
        event_time=ts(2),
        receive_time=ts(3),
        sequence=SequenceRange(
            first=first,
            last=(
                first
                if last is None
                else last
            ),
        ),
        updates=(
            BookUpdate(
                side=Side.BID,
                price=100.0,
                size=2.5,
            ),
        ),
    )


def test_book_level_requires_positive_price() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BookLevel(
            price=0.0,
            size=1.0,
        )


def test_book_level_requires_positive_size() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BookLevel(
            price=100.0,
            size=0.0,
        )


def test_book_update_allows_zero_size_for_deletion() -> None:
    update = BookUpdate(
        side=Side.BID,
        price=100.0,
        size=0.0,
    )

    assert update.size == 0.0


def test_trade_notional() -> None:
    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        trade_id="123",
        event_time=ts(0),
        receive_time=ts(1),
        price=100.0,
        size=2.5,
        aggressor_side=
            AggressorSide.BUY,
    )

    assert trade.notional == 250.0
    assert trade.signed_size == 2.5
    assert trade.signed_notional == 250.0


def test_sell_trade_has_negative_signed_size() -> None:
    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        trade_id="124",
        event_time=ts(0),
        receive_time=ts(1),
        price=100.0,
        size=2.0,
        aggressor_side=
            AggressorSide.SELL,
    )

    assert trade.signed_size == -2.0
    assert trade.signed_notional == -200.0


def test_unknown_aggressor_has_zero_signed_size() -> None:
    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        trade_id="125",
        event_time=ts(0),
        receive_time=ts(1),
        price=100.0,
        size=2.0,
    )

    assert trade.signed_size == 0.0


def test_snapshot_best_prices() -> None:
    book = snapshot()

    assert book.best_bid == 100.0
    assert book.best_ask == 101.0

    assert book.spread == 1.0
    assert book.midpoint == 100.5


def test_snapshot_bid_levels_must_descend() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Snapshot(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(0),
            receive_time=ts(1),
            sequence=100,
            bids=(
                BookLevel(
                    price=99.0,
                    size=1.0,
                ),
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
            ),
            asks=(
                BookLevel(
                    price=101.0,
                    size=1.0,
                ),
            ),
        )


def test_snapshot_ask_levels_must_ascend() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Snapshot(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(0),
            receive_time=ts(1),
            sequence=100,
            bids=(
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
            ),
            asks=(
                BookLevel(
                    price=102.0,
                    size=1.0,
                ),
                BookLevel(
                    price=101.0,
                    size=1.0,
                ),
            ),
        )


def test_snapshot_rejects_duplicate_price_levels() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Snapshot(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(0),
            receive_time=ts(1),
            sequence=100,
            bids=(
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
                BookLevel(
                    price=100.0,
                    size=2.0,
                ),
            ),
            asks=(
                BookLevel(
                    price=101.0,
                    size=1.0,
                ),
            ),
        )


def test_snapshot_rejects_locked_book() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Snapshot(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(0),
            receive_time=ts(1),
            sequence=100,
            bids=(
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
            ),
            asks=(
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
            ),
        )


def test_snapshot_rejects_crossed_book() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Snapshot(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(0),
            receive_time=ts(1),
            sequence=100,
            bids=(
                BookLevel(
                    price=101.0,
                    size=1.0,
                ),
            ),
            asks=(
                BookLevel(
                    price=100.0,
                    size=1.0,
                ),
            ),
        )


def test_sequence_range_contains_values() -> None:
    sequence = SequenceRange(
        first=101,
        last=105,
    )

    assert sequence.contains(101)
    assert sequence.contains(103)
    assert sequence.contains(105)

    assert not sequence.contains(100)
    assert not sequence.contains(106)


def test_sequence_range_rejects_reverse_order() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        SequenceRange(
            first=105,
            last=101,
        )


def test_delta_rejects_duplicate_side_price_updates() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        L2Delta(
            exchange="binance",
            symbol="BTC-USDT",
            event_time=ts(2),
            receive_time=ts(3),
            sequence=SequenceRange(
                first=101,
                last=101,
            ),
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=1.0,
                ),
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=2.0,
                ),
            ),
        )


def test_same_price_can_update_bid_and_ask_in_delta_schema() -> None:
    item = L2Delta(
        exchange="binance",
        symbol="BTC-USDT",
        event_time=ts(2),
        receive_time=ts(3),
        sequence=SequenceRange(
            first=101,
            last=101,
        ),
        updates=(
            BookUpdate(
                side=Side.BID,
                price=100.0,
                size=1.0,
            ),
            BookUpdate(
                side=Side.ASK,
                price=100.0,
                size=2.0,
            ),
        ),
    )

    assert len(item.updates) == 2


def test_sequence_tracker_requires_snapshot() -> None:
    tracker = SequenceTracker()

    with pytest.raises(
        SequenceIntegrityError
    ):
        tracker.validate_and_advance(
            delta(
                first=101,
            )
        )


def test_sequence_tracker_accepts_next_sequence() -> None:
    tracker = SequenceTracker()

    tracker.initialize(
        snapshot(
            sequence=100,
        )
    )

    tracker.validate_and_advance(
        delta(
            first=101,
        )
    )

    assert tracker.last_sequence == 101


def test_sequence_tracker_accepts_range_containing_next_sequence() -> None:
    tracker = SequenceTracker()

    tracker.initialize(
        snapshot(
            sequence=100,
        )
    )

    tracker.validate_and_advance(
        delta(
            first=99,
            last=103,
        )
    )

    assert tracker.last_sequence == 103


def test_sequence_tracker_detects_gap() -> None:
    tracker = SequenceTracker()

    tracker.initialize(
        snapshot(
            sequence=100,
        )
    )

    with pytest.raises(
        SequenceIntegrityError
    ):
        tracker.validate_and_advance(
            delta(
                first=102,
                last=103,
            )
        )


def test_sequence_tracker_detects_stale_delta() -> None:
    tracker = SequenceTracker()

    tracker.initialize(
        snapshot(
            sequence=100,
        )
    )

    with pytest.raises(
        SequenceIntegrityError
    ):
        tracker.validate_and_advance(
            delta(
                first=90,
                last=99,
            )
        )
