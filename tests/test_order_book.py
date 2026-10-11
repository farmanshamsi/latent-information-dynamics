from datetime import datetime, timezone

import pytest

from latent_information_dynamics.market_data.book import (
    L2OrderBook,
    OrderBookStateError,
)
from latent_information_dynamics.market_data.models import (
    BookLevel,
    BookUpdate,
    L2Delta,
    L2Snapshot,
    SequenceIntegrityError,
    SequenceRange,
    Side,
)


UTC = timezone.utc


def ts(
    second: int,
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


def initial_snapshot() -> L2Snapshot:
    return L2Snapshot(
        exchange="binance",
        symbol="BTC-USDT",
        event_time=ts(0),
        receive_time=ts(1),
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
            BookLevel(
                price=98.0,
                size=4.0,
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
            BookLevel(
                price=103.0,
                size=3.0,
            ),
        ),
    )


def make_delta(
    *,
    first: int,
    last: int | None = None,
    updates: tuple[BookUpdate, ...],
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
        updates=updates,
    )


def initialized_book() -> L2OrderBook:
    book = L2OrderBook()

    book.initialize(
        initial_snapshot()
    )

    return book


def test_book_requires_initialization() -> None:
    book = L2OrderBook()

    with pytest.raises(
        OrderBookStateError
    ):
        _ = book.best_bid


def test_snapshot_initializes_book() -> None:
    book = initialized_book()

    assert book.initialized
    assert book.last_sequence == 100

    assert book.best_bid == 100.0
    assert book.best_ask == 101.0


def test_midpoint_and_spread() -> None:
    book = initialized_book()

    assert book.midpoint == 100.5
    assert book.spread == 1.0


def test_top_level_imbalance() -> None:
    book = initialized_book()

    expected = (
        2.0 - 1.0
    ) / (
        2.0 + 1.0
    )

    assert (
        book.top_level_imbalance
        == pytest.approx(
            expected
        )
    )


def test_microprice() -> None:
    book = initialized_book()

    expected = (
        101.0 * 2.0
        + 100.0 * 1.0
    ) / 3.0

    assert (
        book.microprice
        == pytest.approx(
            expected
        )
    )


def test_update_existing_bid_level() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=5.0,
                ),
            ),
        )
    )

    assert book.best_bid == 100.0
    assert book.best_bid_size == 5.0
    assert book.last_sequence == 101


def test_insert_new_best_bid() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.5,
                    size=1.5,
                ),
            ),
        )
    )

    assert book.best_bid == 100.5
    assert book.best_bid_size == 1.5


def test_insert_new_best_ask() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.ASK,
                    price=100.75,
                    size=2.5,
                ),
            ),
        )
    )

    assert book.best_ask == 100.75
    assert book.best_ask_size == 2.5


def test_delete_best_bid_reveals_next_level() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=100.0,
                    size=0.0,
                ),
            ),
        )
    )

    assert book.best_bid == 99.0


def test_delete_best_ask_reveals_next_level() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.ASK,
                    price=101.0,
                    size=0.0,
                ),
            ),
        )
    )

    assert book.best_ask == 102.0


def test_delete_nonexistent_level_is_idempotent() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=97.0,
                    size=0.0,
                ),
            ),
        )
    )

    assert book.best_bid == 100.0


def test_depth_top_two_levels() -> None:
    book = initialized_book()

    bid_depth = book.depth(
        side=Side.BID,
        levels=2,
    )

    ask_depth = book.depth(
        side=Side.ASK,
        levels=2,
    )

    assert bid_depth == 5.0
    assert ask_depth == 3.0


def test_full_depth() -> None:
    book = initialized_book()

    assert (
        book.depth(
            side=Side.BID
        )
        == 9.0
    )

    assert (
        book.depth(
            side=Side.ASK
        )
        == 6.0
    )


def test_depth_imbalance() -> None:
    book = initialized_book()

    expected = (
        5.0 - 3.0
    ) / (
        5.0 + 3.0
    )

    assert (
        book.depth_imbalance(
            levels=2
        )
        == pytest.approx(
            expected
        )
    )


def test_delta_range_advances_sequence() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=99,
            last=104,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=99.0,
                    size=5.0,
                ),
            ),
        )
    )

    assert book.last_sequence == 104


def test_sequence_gap_does_not_mutate_state() -> None:
    book = initialized_book()

    before = book.snapshot()

    with pytest.raises(
        SequenceIntegrityError
    ):
        book.apply(
            make_delta(
                first=102,
                updates=(
                    BookUpdate(
                        side=Side.BID,
                        price=100.0,
                        size=999.0,
                    ),
                ),
            )
        )

    after = book.snapshot()

    assert after == before


def test_crossed_update_is_rejected_and_rolled_back() -> None:
    book = initialized_book()

    before = book.snapshot()

    with pytest.raises(
        OrderBookStateError
    ):
        book.apply(
            make_delta(
                first=101,
                updates=(
                    BookUpdate(
                        side=Side.BID,
                        price=102.0,
                        size=10.0,
                    ),
                ),
            )
        )

    after = book.snapshot()

    assert after == before
    assert book.last_sequence == 100


def test_deleting_all_bids_is_rejected_and_rolled_back() -> None:
    book = initialized_book()

    before = book.snapshot()

    with pytest.raises(
        OrderBookStateError
    ):
        book.apply(
            make_delta(
                first=101,
                updates=(
                    BookUpdate(
                        side=Side.BID,
                        price=100.0,
                        size=0.0,
                    ),
                    BookUpdate(
                        side=Side.BID,
                        price=99.0,
                        size=0.0,
                    ),
                    BookUpdate(
                        side=Side.BID,
                        price=98.0,
                        size=0.0,
                    ),
                ),
            )
        )

    assert book.snapshot() == before


def test_exchange_mismatch_is_rejected() -> None:
    book = initialized_book()

    bad_delta = L2Delta(
        exchange="coinbase",
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
                size=3.0,
            ),
        ),
    )

    with pytest.raises(
        OrderBookStateError
    ):
        book.apply(
            bad_delta
        )


def test_symbol_mismatch_is_rejected() -> None:
    book = initialized_book()

    bad_delta = L2Delta(
        exchange="binance",
        symbol="ETH-USDT",
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
                size=3.0,
            ),
        ),
    )

    with pytest.raises(
        OrderBookStateError
    ):
        book.apply(
            bad_delta
        )


def test_emitted_snapshot_is_canonically_ordered() -> None:
    book = initialized_book()

    book.apply(
        make_delta(
            first=101,
            updates=(
                BookUpdate(
                    side=Side.BID,
                    price=99.5,
                    size=1.0,
                ),
                BookUpdate(
                    side=Side.ASK,
                    price=101.5,
                    size=1.0,
                ),
            ),
        )
    )

    item = book.snapshot()

    bid_prices = [
        level.price
        for level in item.bids
    ]

    ask_prices = [
        level.price
        for level in item.asks
    ]

    assert bid_prices == sorted(
        bid_prices,
        reverse=True,
    )

    assert ask_prices == sorted(
        ask_prices
    )


def test_metrics_object_matches_book_state() -> None:
    book = initialized_book()

    metrics = book.metrics()

    assert metrics.best_bid == 100.0
    assert metrics.best_ask == 101.0
    assert metrics.midpoint == 100.5
    assert metrics.spread == 1.0

    assert (
        metrics.top_level_imbalance
        == pytest.approx(
            1.0 / 3.0
        )
    )
