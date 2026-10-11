from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Iterable

from latent_information_dynamics.data.timestamps import ensure_utc


UTC = timezone.utc


class BinaryTargetError(ValueError):
    """Raised when a binary target or decision window is invalid."""


class ReferencePriceError(ValueError):
    """Raised when reference-price observations are invalid."""


class TieRule(str, Enum):
    """
    Contract convention when final price equals initial price.
    """

    UP = "up"
    DOWN = "down"
    EXCLUDE = "exclude"


@dataclass(frozen=True, slots=True)
class ReferencePriceObservation:
    """
    One observation from the reference price used to define
    the binary outcome.
    """

    source: str
    reference_time: datetime
    available_time: datetime
    price: float

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ReferencePriceError(
                "Reference-price source cannot be empty."
            )

        object.__setattr__(
            self,
            "reference_time",
            ensure_utc(
                self.reference_time,
                field_name="reference_time",
            ),
        )

        object.__setattr__(
            self,
            "available_time",
            ensure_utc(
                self.available_time,
                field_name="available_time",
            ),
        )

        price = float(self.price)

        if not math.isfinite(price):
            raise ReferencePriceError(
                "Reference price must be finite."
            )

        if price <= 0:
            raise ReferencePriceError(
                "Reference price must be positive."
            )

        object.__setattr__(
            self,
            "price",
            price,
        )

    @property
    def latency(self) -> timedelta:
        return (
            self.available_time
            - self.reference_time
        )


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    """
    One binary forecasting interval.
    """

    decision_time: datetime
    resolution_time: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_time",
            ensure_utc(
                self.decision_time,
                field_name="decision_time",
            ),
        )

        object.__setattr__(
            self,
            "resolution_time",
            ensure_utc(
                self.resolution_time,
                field_name="resolution_time",
            ),
        )

        if self.resolution_time <= self.decision_time:
            raise BinaryTargetError(
                "resolution_time must follow decision_time."
            )

    @property
    def horizon(self) -> timedelta:
        return (
            self.resolution_time
            - self.decision_time
        )

    @property
    def feature_cutoff(self) -> datetime:
        return self.decision_time


@dataclass(frozen=True, slots=True)
class BinaryDirectionTarget:
    """
    Realized UP/DOWN target for one completed decision window.
    """

    window: DecisionWindow

    start_reference: ReferencePriceObservation
    end_reference: ReferencePriceObservation

    tie_rule: TieRule
    outcome: int | None

    def __post_init__(self) -> None:
        if (
            self.start_reference.reference_time
            != self.window.decision_time
        ):
            raise BinaryTargetError(
                "Start reference timestamp must equal decision_time."
            )

        if (
            self.end_reference.reference_time
            != self.window.resolution_time
        ):
            raise BinaryTargetError(
                "End reference timestamp must equal resolution_time."
            )

        if (
            self.start_reference.source
            != self.end_reference.source
        ):
            raise BinaryTargetError(
                "Start and end reference prices must use "
                "the same settlement source."
            )

        if self.outcome not in (
            0,
            1,
            None,
        ):
            raise BinaryTargetError(
                "Outcome must be 0, 1 or None."
            )

    @property
    def start_price(self) -> float:
        return self.start_reference.price

    @property
    def end_price(self) -> float:
        return self.end_reference.price

    @property
    def arithmetic_return(self) -> float:
        return (
            self.end_price
            / self.start_price
            - 1.0
        )

    @property
    def log_return(self) -> float:
        return math.log(
            self.end_price
            / self.start_price
        )


@dataclass(frozen=True, slots=True)
class FixedUTCDecisionClock:
    """
    Fixed UTC schedule for six four-hour BTC forecasts per day.
    """

    decision_hours: tuple[int, ...] = (
        0,
        4,
        8,
        12,
        16,
        20,
    )

    horizon: timedelta = timedelta(
        hours=4
    )

    def __post_init__(self) -> None:
        if not self.decision_hours:
            raise BinaryTargetError(
                "At least one decision hour is required."
            )

        if len(
            self.decision_hours
        ) != len(
            set(self.decision_hours)
        ):
            raise BinaryTargetError(
                "Decision hours must be unique."
            )

        for hour in self.decision_hours:
            if hour < 0 or hour > 23:
                raise BinaryTargetError(
                    "Decision hours must be between 0 and 23."
                )

        if tuple(
            sorted(self.decision_hours)
        ) != self.decision_hours:
            raise BinaryTargetError(
                "Decision hours must be sorted."
            )

        if self.horizon <= timedelta(0):
            raise BinaryTargetError(
                "Forecast horizon must be positive."
            )

    def decision_times(
        self,
        day: date,
    ) -> tuple[datetime, ...]:
        return tuple(
            datetime.combine(
                day,
                time(
                    hour=hour,
                    tzinfo=UTC,
                ),
            )
            for hour in self.decision_hours
        )

    def windows(
        self,
        day: date,
    ) -> tuple[DecisionWindow, ...]:
        return tuple(
            DecisionWindow(
                decision_time=decision_time,
                resolution_time=(
                    decision_time
                    + self.horizon
                ),
            )
            for decision_time
            in self.decision_times(day)
        )

    def is_decision_time(
        self,
        value: datetime,
    ) -> bool:
        value = ensure_utc(
            value,
            field_name="decision_time",
        )

        return (
            value.minute == 0
            and value.second == 0
            and value.microsecond == 0
            and value.hour in self.decision_hours
        )

    def next_decision(
        self,
        value: datetime,
    ) -> datetime:
        value = ensure_utc(
            value,
            field_name="value",
        )

        candidates = (
            *self.decision_times(
                value.date()
            ),
            *self.decision_times(
                value.date()
                + timedelta(days=1)
            ),
        )

        for candidate in candidates:
            if candidate >= value:
                return candidate

        raise BinaryTargetError(
            "Unable to determine next decision time."
        )


def build_binary_direction_target(
    *,
    window: DecisionWindow,
    start_reference: ReferencePriceObservation,
    end_reference: ReferencePriceObservation,
    tie_rule: TieRule = TieRule.UP,
) -> BinaryDirectionTarget:
    """
    Construct a realized binary BTC direction label.
    """

    start_price = start_reference.price
    end_price = end_reference.price

    if end_price > start_price:
        outcome: int | None = 1

    elif end_price < start_price:
        outcome = 0

    elif tie_rule is TieRule.UP:
        outcome = 1

    elif tie_rule is TieRule.DOWN:
        outcome = 0

    else:
        outcome = None

    return BinaryDirectionTarget(
        window=window,
        start_reference=start_reference,
        end_reference=end_reference,
        tie_rule=tie_rule,
        outcome=outcome,
    )


def exact_reference_at(
    observations: Iterable[
        ReferencePriceObservation
    ],
    *,
    reference_time: datetime,
    source: str | None = None,
) -> ReferencePriceObservation:
    """
    Retrieve exactly one reference observation for a timestamp.

    Nearest-price substitution is deliberately forbidden.
    """

    reference_time = ensure_utc(
        reference_time,
        field_name="reference_time",
    )

    matches = [
        observation
        for observation in observations
        if (
            observation.reference_time
            == reference_time
            and (
                source is None
                or observation.source == source
            )
        )
    ]

    if not matches:
        raise ReferencePriceError(
            f"No reference price exists at "
            f"{reference_time.isoformat()}."
        )

    if len(matches) > 1:
        raise ReferencePriceError(
            f"Multiple reference prices exist at "
            f"{reference_time.isoformat()}. "
            f"Revision semantics must be resolved explicitly."
        )

    return matches[0]
