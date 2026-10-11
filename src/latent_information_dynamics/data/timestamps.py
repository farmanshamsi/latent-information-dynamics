from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


class TimestampError(ValueError):
    """Raised when a timestamp violates the point-in-time contract."""


def ensure_utc(value: datetime, *, field_name: str) -> datetime:
    """
    Require a timezone-aware datetime and normalize it to UTC.

    Naive datetimes are prohibited because they make cross-source
    synchronization ambiguous.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise TimestampError(
            f"{field_name} must be timezone-aware; received naive datetime "
            f"{value!r}"
        )

    return value.astimezone(timezone.utc)


def _max_timestamp(values: Iterable[datetime | None]) -> datetime:
    usable = [value for value in values if value is not None]

    if not usable:
        raise TimestampError("At least one availability timestamp is required.")

    return max(usable)


@dataclass(frozen=True, slots=True)
class InformationTimestamps:
    """
    Temporal metadata for one observable piece of information.

    Parameters
    ----------
    event_time:
        When the underlying economic or market event occurred.

    publish_time:
        When the source first published the information, if applicable.

    receive_time:
        When our system received the observation.

    process_time:
        When the observation became usable after parsing / transformation.

    source_available_time:
        Optional source-specific availability constraint.

        Examples:
        - blockchain confirmation time
        - official macro release availability
        - vendor dissemination timestamp

    Notes
    -----
    `event_time` is deliberately NOT sufficient for model availability.

    An event may occur before the researcher could have observed it.
    The usable timestamp is therefore derived from dissemination,
    reception and processing times.
    """

    event_time: datetime
    receive_time: datetime
    process_time: datetime

    publish_time: datetime | None = None
    source_available_time: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_time",
            ensure_utc(self.event_time, field_name="event_time"),
        )

        object.__setattr__(
            self,
            "receive_time",
            ensure_utc(self.receive_time, field_name="receive_time"),
        )

        object.__setattr__(
            self,
            "process_time",
            ensure_utc(self.process_time, field_name="process_time"),
        )

        if self.publish_time is not None:
            object.__setattr__(
                self,
                "publish_time",
                ensure_utc(self.publish_time, field_name="publish_time"),
            )

        if self.source_available_time is not None:
            object.__setattr__(
                self,
                "source_available_time",
                ensure_utc(
                    self.source_available_time,
                    field_name="source_available_time",
                ),
            )

    @property
    def available_time(self) -> datetime:
        """
        Earliest timestamp at which the observation is legally usable.

        Availability is constrained by every applicable dissemination
        and processing timestamp.
        """
        return _max_timestamp(
            (
                self.publish_time,
                self.receive_time,
                self.process_time,
                self.source_available_time,
            )
        )

    def is_available_by(self, prediction_time: datetime) -> bool:
        """
        Return True only if the observation was available by prediction_time.
        """
        prediction_time = ensure_utc(
            prediction_time,
            field_name="prediction_time",
        )

        return self.available_time <= prediction_time

    def latency_seconds(self) -> float:
        """
        Total elapsed time between the underlying event and model availability.

        This is observational latency, not necessarily causal price-response
        latency.
        """
        return (self.available_time - self.event_time).total_seconds()

    def audit(self) -> tuple[str, ...]:
        """
        Return suspicious temporal relationships without automatically
        rejecting the record.

        Real financial feeds can contain clock skew, vendor corrections,
        exchange-clock inconsistencies and delayed timestamps, so these
        relationships should be audited rather than blindly assumed.
        """
        violations: list[str] = []

        if self.publish_time is not None:
            if self.publish_time < self.event_time:
                violations.append("publish_time precedes event_time")

            if self.receive_time < self.publish_time:
                violations.append("receive_time precedes publish_time")

        if self.process_time < self.receive_time:
            violations.append("process_time precedes receive_time")

        if (
            self.source_available_time is not None
            and self.source_available_time < self.event_time
        ):
            violations.append(
                "source_available_time precedes event_time"
            )

        if self.available_time < self.event_time:
            violations.append("available_time precedes event_time")

        return tuple(violations)
