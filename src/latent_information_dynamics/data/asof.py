from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Literal

from .contracts import FeatureObservation
from .timestamps import ensure_utc


AgeBasis = Literal["available", "event"]


class MissingRequiredFeature(RuntimeError):
    """Raised when a required feature is unavailable at prediction time."""


class AmbiguousObservationError(RuntimeError):
    """
    Raised when multiple observations are equally eligible for the
    same as-of request.

    Silently selecting one would make dataset construction dependent
    on input row ordering.
    """


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """
    Specification for one point-in-time feature lookup.

    Parameters
    ----------
    name:
        FeatureObservation.name to retrieve.

    output_name:
        Optional alias in the resulting feature vector.

    source:
        Restrict lookup to one source.

    entity:
        Restrict lookup to one entity, symbol or instrument.

    required:
        If True, absence of a legal observation fails the join.

    max_age:
        Optional maximum permitted age.

    age_basis:
        Whether age is measured from model availability time or
        underlying economic event time.

        "available" is the safe default for most information feeds.
    """

    name: str
    output_name: str | None = None
    source: str | None = None
    entity: str | None = None
    required: bool = True
    max_age: timedelta | None = None
    age_basis: AgeBasis = "available"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature request name cannot be empty.")

        if self.output_name is not None:
            if not self.output_name.strip():
                raise ValueError(
                    "output_name cannot be empty when supplied."
                )

        if self.source is not None:
            if not self.source.strip():
                raise ValueError(
                    "source cannot be empty when supplied."
                )

        if self.entity is not None:
            if not self.entity.strip():
                raise ValueError(
                    "entity cannot be empty when supplied."
                )

        if self.max_age is not None:
            if self.max_age < timedelta(0):
                raise ValueError(
                    "max_age cannot be negative."
                )

        if self.age_basis not in ("available", "event"):
            raise ValueError(
                "age_basis must be 'available' or 'event'."
            )

    @property
    def key(self) -> str:
        return self.output_name or self.name


@dataclass(frozen=True, slots=True)
class JoinedFeature:
    """
    One feature selected by the as-of engine.
    """

    request: FeatureRequest
    observation: FeatureObservation[Any]
    prediction_time: datetime

    @property
    def key(self) -> str:
        return self.request.key

    @property
    def value(self) -> Any:
        return self.observation.value

    @property
    def availability_age(self) -> timedelta:
        return (
            self.prediction_time
            - self.observation.available_time
        )

    @property
    def event_age(self) -> timedelta:
        return (
            self.prediction_time
            - self.observation.timestamps.event_time
        )


@dataclass(frozen=True, slots=True)
class AsOfJoinResult:
    """
    Immutable result of one feature-vector construction.
    """

    prediction_time: datetime
    features: tuple[JoinedFeature, ...]
    missing_optional: tuple[str, ...]

    def values(self) -> dict[str, Any]:
        return {
            feature.key: feature.value
            for feature in self.features
        }

    def observations(
        self,
    ) -> dict[str, FeatureObservation[Any]]:
        return {
            feature.key: feature.observation
            for feature in self.features
        }

    def availability_ages(
        self,
    ) -> dict[str, timedelta]:
        return {
            feature.key: feature.availability_age
            for feature in self.features
        }

    def event_ages(
        self,
    ) -> dict[str, timedelta]:
        return {
            feature.key: feature.event_age
            for feature in self.features
        }


def _matches_request(
    observation: FeatureObservation[Any],
    request: FeatureRequest,
) -> bool:
    if observation.name != request.name:
        return False

    if request.source is not None:
        if observation.source != request.source:
            return False

    if request.entity is not None:
        if observation.entity != request.entity:
            return False

    return True


def _age(
    observation: FeatureObservation[Any],
    prediction_time: datetime,
    basis: AgeBasis,
) -> timedelta:
    if basis == "available":
        return (
            prediction_time
            - observation.available_time
        )

    return (
        prediction_time
        - observation.timestamps.event_time
    )


def select_asof(
    observations: Iterable[FeatureObservation[Any]],
    request: FeatureRequest,
    prediction_time: datetime,
) -> FeatureObservation[Any] | None:
    """
    Select the latest legally available observation satisfying a request.

    Selection is performed using `available_time`, never merely event_time.
    """
    prediction_time = ensure_utc(
        prediction_time,
        field_name="prediction_time",
    )

    eligible: list[FeatureObservation[Any]] = []

    for observation in observations:
        if not _matches_request(
            observation,
            request,
        ):
            continue

        if observation.available_time > prediction_time:
            continue

        if request.max_age is not None:
            observation_age = _age(
                observation,
                prediction_time,
                request.age_basis,
            )

            # Negative age means the selected age clock lies
            # in the future relative to the forecast origin.
            if observation_age < timedelta(0):
                continue

            if observation_age > request.max_age:
                continue

        eligible.append(observation)

    if not eligible:
        return None

    latest_time = max(
        observation.available_time
        for observation in eligible
    )

    latest = [
        observation
        for observation in eligible
        if observation.available_time == latest_time
    ]

    if len(latest) > 1:
        raise AmbiguousObservationError(
            f"Multiple observations for feature "
            f"{request.name!r} are available at the same "
            f"latest timestamp {latest_time.isoformat()}. "
            f"Resolve duplicate/revision semantics explicitly."
        )

    return latest[0]


def asof_join(
    observations: Iterable[FeatureObservation[Any]],
    requests: Iterable[FeatureRequest],
    prediction_time: datetime,
) -> AsOfJoinResult:
    """
    Construct one point-in-time-safe feature vector.

    Required features fail loudly when unavailable.
    Optional features are recorded as missing.
    """
    prediction_time = ensure_utc(
        prediction_time,
        field_name="prediction_time",
    )

    observation_pool = tuple(observations)
    request_set = tuple(requests)

    keys = [
        request.key
        for request in request_set
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Feature request output keys must be unique."
        )

    joined: list[JoinedFeature] = []
    missing_optional: list[str] = []
    missing_required: list[str] = []

    for request in request_set:
        selected = select_asof(
            observation_pool,
            request,
            prediction_time,
        )

        if selected is None:
            if request.required:
                missing_required.append(
                    request.key
                )
            else:
                missing_optional.append(
                    request.key
                )

            continue

        joined.append(
            JoinedFeature(
                request=request,
                observation=selected,
                prediction_time=prediction_time,
            )
        )

    if missing_required:
        missing_text = ", ".join(
            repr(name)
            for name in missing_required
        )

        raise MissingRequiredFeature(
            f"Required feature(s) unavailable at "
            f"{prediction_time.isoformat()}: "
            f"{missing_text}"
        )

    return AsOfJoinResult(
        prediction_time=prediction_time,
        features=tuple(joined),
        missing_optional=tuple(missing_optional),
    )
