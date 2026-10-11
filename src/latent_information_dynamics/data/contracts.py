from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Generic, TypeVar

from .timestamps import InformationTimestamps, ensure_utc


T = TypeVar("T")


class PointInTimeViolation(RuntimeError):
    """
    Raised when information is accessed before it was actually available.
    """


@dataclass(frozen=True, slots=True)
class FeatureObservation(Generic[T]):
    """
    One point-in-time-safe feature observation.

    A feature value is inseparable from the timestamp at which it became
    available to the research system.
    """

    name: str
    value: T
    source: str
    timestamps: InformationTimestamps
    entity: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature name cannot be empty.")

        if not self.source.strip():
            raise ValueError("Feature source cannot be empty.")

    @property
    def available_time(self) -> datetime:
        return self.timestamps.available_time

    def is_available_by(self, prediction_time: datetime) -> bool:
        return self.timestamps.is_available_by(prediction_time)

    def require_available_by(
        self,
        prediction_time: datetime,
    ) -> FeatureObservation[T]:
        """
        Enforce the no-look-ahead rule.

        Returns self when legal so the method can be inserted naturally
        into data pipelines.
        """
        prediction_time = ensure_utc(
            prediction_time,
            field_name="prediction_time",
        )

        if not self.is_available_by(prediction_time):
            raise PointInTimeViolation(
                f"Feature {self.name!r} from {self.source!r} "
                f"was not available at prediction time "
                f"{prediction_time.isoformat()}. "
                f"Available only from "
                f"{self.available_time.isoformat()}."
            )

        return self


@dataclass(frozen=True, slots=True)
class NumericFeatureObservation(FeatureObservation[float]):
    """
    Numeric feature with finite-value validation.
    """

    def __post_init__(self) -> None:
        FeatureObservation.__post_init__(self)

        if not isinstance(self.value, Real):
            raise TypeError(
                f"{self.name!r} must contain a numeric value."
            )

        value = float(self.value)

        if value != value:
            raise ValueError(
                f"{self.name!r} cannot contain NaN."
            )

        if value in (float("inf"), float("-inf")):
            raise ValueError(
                f"{self.name!r} cannot contain infinity."
            )


def latest_available(
    observations: list[FeatureObservation[T]],
    prediction_time: datetime,
) -> FeatureObservation[T] | None:
    """
    Return the most recently available observation at prediction_time.

    Future observations are excluded before selection.

    This is the primitive underlying safe as-of joins.
    """
    prediction_time = ensure_utc(
        prediction_time,
        field_name="prediction_time",
    )

    eligible = [
        obs
        for obs in observations
        if obs.available_time <= prediction_time
    ]

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda obs: obs.available_time,
    )
