from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from .models import (
    BookLevel,
    L2Delta,
    L2Snapshot,
    MarketDataValidationError,
    SequenceIntegrityError,
    SequenceTracker,
    Side,
)


class OrderBookStateError(RuntimeError):
    """Raised when reconstructed order-book state becomes invalid."""


@dataclass(frozen=True, slots=True)
class BookMetrics:
    """
    Top-of-book statistics derived from reconstructed state.
    """

    best_bid: float
    best_ask: float

    best_bid_size: float
    best_ask_size: float

    midpoint: float
    spread: float

    top_level_imbalance: float
    microprice: float


class L2OrderBook:
    """
    Stateful canonical Level-2 order book.

    The internal representation uses price -> absolute size maps.

    Public snapshots are emitted as immutable, canonically ordered
    L2Snapshot objects.
    """

    def __init__(self) -> None:
        self._exchange: str | None = None
        self._symbol: str | None = None

        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

        self._event_time: datetime | None = None
        self._receive_time: datetime | None = None

        self._sequence = SequenceTracker()

    @property
    def initialized(self) -> bool:
        return self._sequence.last_sequence is not None

    @property
    def exchange(self) -> str:
        if self._exchange is None:
            raise OrderBookStateError(
                "Order book has not been initialized."
            )

        return self._exchange

    @property
    def symbol(self) -> str:
        if self._symbol is None:
            raise OrderBookStateError(
                "Order book has not been initialized."
            )

        return self._symbol

    @property
    def last_sequence(self) -> int:
        if self._sequence.last_sequence is None:
            raise OrderBookStateError(
                "Order book has not been initialized."
            )

        return self._sequence.last_sequence

    def initialize(
        self,
        snapshot: L2Snapshot,
    ) -> None:
        """
        Replace all existing state with a trusted snapshot.
        """
        self._exchange = snapshot.exchange
        self._symbol = snapshot.symbol

        self._bids = {
            level.price: level.size
            for level in snapshot.bids
        }

        self._asks = {
            level.price: level.size
            for level in snapshot.asks
        }

        self._event_time = snapshot.event_time
        self._receive_time = snapshot.receive_time

        self._sequence.initialize(
            snapshot
        )

        self._validate_state()

    def _require_initialized(
        self,
    ) -> None:
        if not self.initialized:
            raise OrderBookStateError(
                "Order book must be initialized from a snapshot."
            )

    def _validate_instrument(
        self,
        delta: L2Delta,
    ) -> None:
        if delta.exchange != self.exchange:
            raise OrderBookStateError(
                f"Exchange mismatch: book={self.exchange!r}, "
                f"delta={delta.exchange!r}."
            )

        if delta.symbol != self.symbol:
            raise OrderBookStateError(
                f"Symbol mismatch: book={self.symbol!r}, "
                f"delta={delta.symbol!r}."
            )

    def _validate_state(
        self,
    ) -> None:
        if not self._bids:
            raise OrderBookStateError(
                "Reconstructed book has no bid levels."
            )

        if not self._asks:
            raise OrderBookStateError(
                "Reconstructed book has no ask levels."
            )

        for side_name, levels in (
            ("bid", self._bids),
            ("ask", self._asks),
        ):
            for price, size in levels.items():
                if not isfinite(price) or price <= 0:
                    raise OrderBookStateError(
                        f"Invalid {side_name} price {price!r}."
                    )

                if not isfinite(size) or size <= 0:
                    raise OrderBookStateError(
                        f"Invalid {side_name} size {size!r}."
                    )

        if self.best_bid >= self.best_ask:
            raise OrderBookStateError(
                "Reconstructed book is locked or crossed: "
                f"best_bid={self.best_bid}, "
                f"best_ask={self.best_ask}."
            )

    def _apply_update(
        self,
        *,
        side: Side,
        price: float,
        size: float,
    ) -> None:
        levels = (
            self._bids
            if side is Side.BID
            else self._asks
        )

        if size == 0.0:
            levels.pop(
                price,
                None,
            )
            return

        levels[price] = size

    def apply(
        self,
        delta: L2Delta,
    ) -> None:
        """
        Validate and atomically apply one depth update.

        If the resulting state is invalid, the mutation is rolled back.
        """
        self._require_initialized()
        self._validate_instrument(
            delta
        )

        previous_bids = self._bids.copy()
        previous_asks = self._asks.copy()
        previous_event_time = self._event_time
        previous_receive_time = self._receive_time
        previous_sequence = self.last_sequence

        try:
            self._sequence.validate_and_advance(
                delta
            )

            for update in delta.updates:
                self._apply_update(
                    side=update.side,
                    price=update.price,
                    size=update.size,
                )

            self._event_time = delta.event_time
            self._receive_time = delta.receive_time

            self._validate_state()

        except Exception:
            self._bids = previous_bids
            self._asks = previous_asks

            self._event_time = previous_event_time
            self._receive_time = previous_receive_time

            self._sequence.last_sequence = (
                previous_sequence
            )

            raise

    @property
    def best_bid(self) -> float:
        self._require_initialized()

        return max(
            self._bids
        )

    @property
    def best_ask(self) -> float:
        self._require_initialized()

        return min(
            self._asks
        )

    @property
    def best_bid_size(self) -> float:
        return self._bids[
            self.best_bid
        ]

    @property
    def best_ask_size(self) -> float:
        return self._asks[
            self.best_ask
        ]

    @property
    def midpoint(self) -> float:
        return (
            self.best_bid
            + self.best_ask
        ) / 2.0

    @property
    def spread(self) -> float:
        return (
            self.best_ask
            - self.best_bid
        )

    @property
    def top_level_imbalance(self) -> float:
        """
        Top-of-book queue imbalance:

            (Q_bid - Q_ask) / (Q_bid + Q_ask)
        """
        bid_size = self.best_bid_size
        ask_size = self.best_ask_size

        denominator = (
            bid_size
            + ask_size
        )

        if denominator <= 0:
            raise OrderBookStateError(
                "Top-level depth denominator must be positive."
            )

        return (
            bid_size
            - ask_size
        ) / denominator

    @property
    def microprice(self) -> float:
        """
        Queue-weighted microprice:

            (ask * bid_size + bid * ask_size)
            --------------------------------
                 bid_size + ask_size
        """
        bid = self.best_bid
        ask = self.best_ask

        bid_size = self.best_bid_size
        ask_size = self.best_ask_size

        denominator = (
            bid_size
            + ask_size
        )

        if denominator <= 0:
            raise OrderBookStateError(
                "Microprice denominator must be positive."
            )

        return (
            ask * bid_size
            + bid * ask_size
        ) / denominator

    def depth(
        self,
        *,
        side: Side,
        levels: int | None = None,
    ) -> float:
        """
        Sum displayed quantity over the first N price levels.

        If levels is None, sum the entire reconstructed side.
        """
        self._require_initialized()

        if levels is not None and levels <= 0:
            raise ValueError(
                "levels must be positive when supplied."
            )

        if side is Side.BID:
            ordered = sorted(
                self._bids.items(),
                reverse=True,
            )
        else:
            ordered = sorted(
                self._asks.items()
            )

        if levels is not None:
            ordered = ordered[
                :levels
            ]

        return sum(
            size
            for _, size in ordered
        )

    def depth_imbalance(
        self,
        *,
        levels: int | None = None,
    ) -> float:
        """
        Multi-level displayed-depth imbalance:

            (D_bid - D_ask) / (D_bid + D_ask)
        """
        bid_depth = self.depth(
            side=Side.BID,
            levels=levels,
        )

        ask_depth = self.depth(
            side=Side.ASK,
            levels=levels,
        )

        denominator = (
            bid_depth
            + ask_depth
        )

        if denominator <= 0:
            raise OrderBookStateError(
                "Depth imbalance denominator must be positive."
            )

        return (
            bid_depth
            - ask_depth
        ) / denominator

    def metrics(
        self,
    ) -> BookMetrics:
        return BookMetrics(
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            best_bid_size=
                self.best_bid_size,
            best_ask_size=
                self.best_ask_size,
            midpoint=self.midpoint,
            spread=self.spread,
            top_level_imbalance=
                self.top_level_imbalance,
            microprice=self.microprice,
        )

    def snapshot(
        self,
    ) -> L2Snapshot:
        """
        Emit current reconstructed state as an immutable snapshot.
        """
        self._require_initialized()

        if (
            self._event_time is None
            or self._receive_time is None
        ):
            raise OrderBookStateError(
                "Book timestamps are unavailable."
            )

        bids = tuple(
            BookLevel(
                price=price,
                size=size,
            )
            for price, size
            in sorted(
                self._bids.items(),
                reverse=True,
            )
        )

        asks = tuple(
            BookLevel(
                price=price,
                size=size,
            )
            for price, size
            in sorted(
                self._asks.items()
            )
        )

        return L2Snapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            event_time=self._event_time,
            receive_time=self._receive_time,
            sequence=self.last_sequence,
            bids=bids,
            asks=asks,
        )
