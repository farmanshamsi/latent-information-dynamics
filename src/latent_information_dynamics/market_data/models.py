from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Iterable

from latent_information_dynamics.data.timestamps import ensure_utc


class MarketDataValidationError(ValueError):
    """Raised when normalized market data violate canonical invariants."""


class SequenceIntegrityError(RuntimeError):
    """Raised when market-data sequence continuity is broken."""


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class AggressorSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


def _positive_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = float(value)

    if not isfinite(numeric):
        raise MarketDataValidationError(
            f"{field_name} must be finite."
        )

    if numeric <= 0:
        raise MarketDataValidationError(
            f"{field_name} must be positive."
        )

    return numeric


def _nonnegative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    numeric = float(value)

    if not isfinite(numeric):
        raise MarketDataValidationError(
            f"{field_name} must be finite."
        )

    if numeric < 0:
        raise MarketDataValidationError(
            f"{field_name} cannot be negative."
        )

    return numeric


def _nonempty(
    value: str,
    *,
    field_name: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise MarketDataValidationError(
            f"{field_name} cannot be empty."
        )

    return normalized


@dataclass(frozen=True, slots=True)
class BookLevel:
    """
    One price level in a normalized order book.

    Snapshot levels require strictly positive size.

    Delta updates use BookUpdate because zero quantity has
    deletion semantics.
    """

    price: float
    size: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "price",
            _positive_finite(
                self.price,
                field_name="price",
            ),
        )

        object.__setattr__(
            self,
            "size",
            _positive_finite(
                self.size,
                field_name="size",
            ),
        )


@dataclass(frozen=True, slots=True)
class BookUpdate:
    """
    One normalized L2 update.

    `size == 0` means remove the price level.
    Otherwise `size` is the new absolute quantity at that level.

    The canonical representation intentionally uses absolute
    quantities rather than vendor-specific quantity deltas.
    """

    side: Side
    price: float
    size: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "price",
            _positive_finite(
                self.price,
                field_name="price",
            ),
        )

        object.__setattr__(
            self,
            "size",
            _nonnegative_finite(
                self.size,
                field_name="size",
            ),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """
    One normalized executed trade.
    """

    exchange: str
    symbol: str

    trade_id: str

    event_time: datetime
    receive_time: datetime

    price: float
    size: float

    aggressor_side: AggressorSide = AggressorSide.UNKNOWN

    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exchange",
            _nonempty(
                self.exchange,
                field_name="exchange",
            ),
        )

        object.__setattr__(
            self,
            "symbol",
            _nonempty(
                self.symbol,
                field_name="symbol",
            ),
        )

        object.__setattr__(
            self,
            "trade_id",
            _nonempty(
                self.trade_id,
                field_name="trade_id",
            ),
        )

        object.__setattr__(
            self,
            "event_time",
            ensure_utc(
                self.event_time,
                field_name="event_time",
            ),
        )

        object.__setattr__(
            self,
            "receive_time",
            ensure_utc(
                self.receive_time,
                field_name="receive_time",
            ),
        )

        object.__setattr__(
            self,
            "price",
            _positive_finite(
                self.price,
                field_name="price",
            ),
        )

        object.__setattr__(
            self,
            "size",
            _positive_finite(
                self.size,
                field_name="size",
            ),
        )

        if (
            self.sequence is not None
            and self.sequence < 0
        ):
            raise MarketDataValidationError(
                "sequence cannot be negative."
            )

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def signed_size(self) -> float:
        if self.aggressor_side is AggressorSide.BUY:
            return self.size

        if self.aggressor_side is AggressorSide.SELL:
            return -self.size

        return 0.0

    @property
    def signed_notional(self) -> float:
        return self.price * self.signed_size


@dataclass(frozen=True, slots=True)
class SequenceRange:
    """
    Inclusive sequence-number range represented by one message.

    Some exchanges publish one number per message.
    Others publish update ranges such as [U, u].

    Canonical normalization supports both.
    """

    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first < 0:
            raise MarketDataValidationError(
                "first sequence cannot be negative."
            )

        if self.last < self.first:
            raise MarketDataValidationError(
                "last sequence cannot precede first sequence."
            )

    def contains(
        self,
        sequence: int,
    ) -> bool:
        return (
            self.first
            <= sequence
            <= self.last
        )


def _validate_levels(
    levels: tuple[BookLevel, ...],
    *,
    side: Side,
) -> None:
    if not levels:
        raise MarketDataValidationError(
            f"{side.value} levels cannot be empty."
        )

    prices = [
        level.price
        for level in levels
    ]

    if len(prices) != len(set(prices)):
        raise MarketDataValidationError(
            f"{side.value} prices must be unique."
        )

    if side is Side.BID:
        expected = sorted(
            prices,
            reverse=True,
        )
    else:
        expected = sorted(
            prices,
        )

    if prices != expected:
        raise MarketDataValidationError(
            f"{side.value} levels are not "
            f"in canonical price order."
        )


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    """
    Canonical Level-2 order-book snapshot.

    Bids must be ordered highest to lowest.
    Asks must be ordered lowest to highest.
    """

    exchange: str
    symbol: str

    event_time: datetime
    receive_time: datetime

    sequence: int

    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exchange",
            _nonempty(
                self.exchange,
                field_name="exchange",
            ),
        )

        object.__setattr__(
            self,
            "symbol",
            _nonempty(
                self.symbol,
                field_name="symbol",
            ),
        )

        object.__setattr__(
            self,
            "event_time",
            ensure_utc(
                self.event_time,
                field_name="event_time",
            ),
        )

        object.__setattr__(
            self,
            "receive_time",
            ensure_utc(
                self.receive_time,
                field_name="receive_time",
            ),
        )

        if self.sequence < 0:
            raise MarketDataValidationError(
                "sequence cannot be negative."
            )

        _validate_levels(
            self.bids,
            side=Side.BID,
        )

        _validate_levels(
            self.asks,
            side=Side.ASK,
        )

        if self.best_bid >= self.best_ask:
            raise MarketDataValidationError(
                "Snapshot is crossed or locked: "
                f"best_bid={self.best_bid}, "
                f"best_ask={self.best_ask}."
            )

    @property
    def best_bid(self) -> float:
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        return self.asks[0].price

    @property
    def best_bid_size(self) -> float:
        return self.bids[0].size

    @property
    def best_ask_size(self) -> float:
        return self.asks[0].size

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float:
        return (
            self.best_bid
            + self.best_ask
        ) / 2.0


@dataclass(frozen=True, slots=True)
class L2Delta:
    """
    Canonical Level-2 order-book update.

    Updates contain absolute new quantities.

    `size == 0` removes the corresponding level.
    """

    exchange: str
    symbol: str

    event_time: datetime
    receive_time: datetime

    sequence: SequenceRange

    updates: tuple[BookUpdate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exchange",
            _nonempty(
                self.exchange,
                field_name="exchange",
            ),
        )

        object.__setattr__(
            self,
            "symbol",
            _nonempty(
                self.symbol,
                field_name="symbol",
            ),
        )

        object.__setattr__(
            self,
            "event_time",
            ensure_utc(
                self.event_time,
                field_name="event_time",
            ),
        )

        object.__setattr__(
            self,
            "receive_time",
            ensure_utc(
                self.receive_time,
                field_name="receive_time",
            ),
        )

        if not self.updates:
            raise MarketDataValidationError(
                "L2 delta must contain at least one update."
            )

        keys = [
            (
                update.side,
                update.price,
            )
            for update in self.updates
        ]

        if len(keys) != len(set(keys)):
            raise MarketDataValidationError(
                "An L2 delta cannot update the same "
                "side/price level more than once."
            )


@dataclass(slots=True)
class SequenceTracker:
    """
    Validate normalized depth-stream continuity.

    After a snapshot with sequence S, the next delta must contain
    sequence S + 1.

    A delta may span several vendor sequence numbers.
    """

    last_sequence: int | None = None

    def initialize(
        self,
        snapshot: L2Snapshot,
    ) -> None:
        self.last_sequence = snapshot.sequence

    def validate_and_advance(
        self,
        delta: L2Delta,
    ) -> None:
        if self.last_sequence is None:
            raise SequenceIntegrityError(
                "SequenceTracker must be initialized "
                "from a snapshot before applying deltas."
            )

        expected = self.last_sequence + 1

        if not delta.sequence.contains(
            expected
        ):
            raise SequenceIntegrityError(
                f"Order-book sequence gap: expected "
                f"{expected}, received range "
                f"[{delta.sequence.first}, "
                f"{delta.sequence.last}]."
            )

        self.last_sequence = (
            delta.sequence.last
        )
