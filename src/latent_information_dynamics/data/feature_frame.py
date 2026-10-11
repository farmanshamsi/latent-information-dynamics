from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Any, Callable, Iterable, Mapping

from .asof import (
    AsOfJoinResult,
    FeatureRequest,
    asof_join,
)
from .contracts import FeatureObservation
from .provenance import (
    FeatureProvenance,
    RawObservationRef,
    TransformationRecord,
)
from .timestamps import ensure_utc


class FeatureFrameError(ValueError):
    """Raised when a model feature frame is invalid."""


@dataclass(frozen=True, slots=True)
class ModelFeature:
    """
    One model-ready feature plus complete lineage.
    """

    name: str
    value: Any
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise FeatureFrameError(
                "Model feature name cannot be empty."
            )

        if self.name != self.provenance.output_name:
            raise FeatureFrameError(
                "Model feature name must match provenance output_name."
            )

    @property
    def available_time(self) -> datetime:
        return self.provenance.available_time


@dataclass(frozen=True, slots=True)
class FeatureFrameRow:
    """
    Immutable model feature vector for one prediction timestamp.
    """

    prediction_time: datetime
    features: tuple[ModelFeature, ...]
    missing_optional: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prediction_time",
            ensure_utc(
                self.prediction_time,
                field_name="prediction_time",
            ),
        )

        names = [
            feature.name
            for feature in self.features
        ]

        if len(names) != len(set(names)):
            raise FeatureFrameError(
                "Feature names must be unique within a frame."
            )

        for feature in self.features:
            if feature.available_time > self.prediction_time:
                raise FeatureFrameError(
                    f"Feature {feature.name!r} contains "
                    f"future information."
                )

    def values(self) -> dict[str, Any]:
        return {
            feature.name: feature.value
            for feature in self.features
        }

    def provenance(self) -> dict[str, FeatureProvenance]:
        return {
            feature.name: feature.provenance
            for feature in self.features
        }

    def get(
        self,
        name: str,
    ) -> ModelFeature:
        for feature in self.features:
            if feature.name == name:
                return feature

        raise KeyError(
            f"Feature {name!r} is not present in the frame."
        )

    def with_feature(
        self,
        feature: ModelFeature,
    ) -> "FeatureFrameRow":
        if feature.name in self.values():
            raise FeatureFrameError(
                f"Feature {feature.name!r} already exists."
            )

        return FeatureFrameRow(
            prediction_time=self.prediction_time,
            features=(
                *self.features,
                feature,
            ),
            missing_optional=self.missing_optional,
        )


def from_asof_result(
    result: AsOfJoinResult,
) -> FeatureFrameRow:
    """
    Convert point-in-time selections into auditable model features.
    """
    features: list[ModelFeature] = []

    for joined in result.features:
        raw_ref = RawObservationRef.from_observation(
            key=joined.key,
            observation=joined.observation,
        )

        provenance = FeatureProvenance(
            output_name=joined.key,
            prediction_time=result.prediction_time,
            raw_inputs=(raw_ref,),
            transformation=TransformationRecord.from_mapping(
                name="asof_select",
                version="1.0.0",
            ),
        )

        features.append(
            ModelFeature(
                name=joined.key,
                value=joined.value,
                provenance=provenance,
            )
        )

    return FeatureFrameRow(
        prediction_time=result.prediction_time,
        features=tuple(features),
        missing_optional=result.missing_optional,
    )


def build_feature_frame(
    observations: Iterable[FeatureObservation[Any]],
    requests: Iterable[FeatureRequest],
    prediction_time: datetime,
) -> FeatureFrameRow:
    """
    Perform the point-in-time join and return an auditable feature row.
    """
    joined = asof_join(
        observations=observations,
        requests=requests,
        prediction_time=prediction_time,
    )

    return from_asof_result(
        joined
    )


def _deduplicate_raw_inputs(
    features: Iterable[ModelFeature],
) -> tuple[RawObservationRef, ...]:
    output: list[RawObservationRef] = []
    seen: set[
        tuple[
            str,
            str,
            str,
            str | None,
            datetime,
            datetime,
        ]
    ] = set()

    for feature in features:
        for raw in feature.provenance.raw_inputs:
            identity = (
                raw.key,
                raw.observation_name,
                raw.source,
                raw.entity,
                raw.event_time,
                raw.available_time,
            )

            if identity in seen:
                continue

            seen.add(identity)
            output.append(raw)

    return tuple(output)


def derive_numeric_feature(
    row: FeatureFrameRow,
    *,
    output_name: str,
    input_names: tuple[str, ...],
    function: Callable[..., Real],
    transformation_name: str,
    transformation_version: str,
    parameters: Mapping[str, Any] | None = None,
) -> FeatureFrameRow:
    """
    Derive one numeric feature while retaining transitive lineage.

    `function` receives values in exactly the order specified by
    `input_names`.
    """
    if not input_names:
        raise FeatureFrameError(
            "Derived feature requires at least one input."
        )

    upstream = tuple(
        row.get(name)
        for name in input_names
    )

    values = [
        feature.value
        for feature in upstream
    ]

    value = function(
        *values
    )

    if not isinstance(
        value,
        Real,
    ):
        raise FeatureFrameError(
            "Derived numeric feature must return a real number."
        )

    numeric_value = float(value)

    if not math.isfinite(
        numeric_value
    ):
        raise FeatureFrameError(
            "Derived numeric feature must be finite."
        )

    raw_inputs = _deduplicate_raw_inputs(
        upstream
    )

    provenance = FeatureProvenance(
        output_name=output_name,
        prediction_time=row.prediction_time,
        raw_inputs=raw_inputs,
        transformation=TransformationRecord.from_mapping(
            name=transformation_name,
            version=transformation_version,
            parameters=parameters,
        ),
        upstream_feature_fingerprints=tuple(
            feature.provenance.fingerprint
            for feature in upstream
        ),
    )

    derived = ModelFeature(
        name=output_name,
        value=numeric_value,
        provenance=provenance,
    )

    return row.with_feature(
        derived
    )
