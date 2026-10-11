from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .contracts import FeatureObservation
from .timestamps import ensure_utc


class ProvenanceError(ValueError):
    """Raised when feature provenance metadata is invalid."""


def _iso(value: datetime) -> str:
    return ensure_utc(
        value,
        field_name="provenance_timestamp",
    ).isoformat()


def _canonical_json(
    value: Mapping[str, Any],
) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError as exc:
        raise ProvenanceError(
            "Transformation parameters must be JSON serializable."
        ) from exc


@dataclass(frozen=True, slots=True)
class RawObservationRef:
    """
    Immutable reference to one raw observation contributing to a
    model feature.
    """

    key: str
    observation_name: str
    source: str
    entity: str | None
    event_time: datetime
    available_time: datetime

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ProvenanceError(
                "Raw observation key cannot be empty."
            )

        if not self.observation_name.strip():
            raise ProvenanceError(
                "observation_name cannot be empty."
            )

        if not self.source.strip():
            raise ProvenanceError(
                "source cannot be empty."
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
            "available_time",
            ensure_utc(
                self.available_time,
                field_name="available_time",
            ),
        )

    @classmethod
    def from_observation(
        cls,
        *,
        key: str,
        observation: FeatureObservation[Any],
    ) -> "RawObservationRef":
        return cls(
            key=key,
            observation_name=observation.name,
            source=observation.source,
            entity=observation.entity,
            event_time=observation.timestamps.event_time,
            available_time=observation.available_time,
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "available_time": _iso(
                self.available_time
            ),
            "entity": self.entity,
            "event_time": _iso(
                self.event_time
            ),
            "key": self.key,
            "observation_name": self.observation_name,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """
    Reproducible description of one feature transformation.
    """

    name: str
    version: str
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProvenanceError(
                "Transformation name cannot be empty."
            )

        if not self.version.strip():
            raise ProvenanceError(
                "Transformation version cannot be empty."
            )

        try:
            parsed = json.loads(
                self.parameters_json
            )
        except json.JSONDecodeError as exc:
            raise ProvenanceError(
                "parameters_json must contain valid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise ProvenanceError(
                "Transformation parameters must be a JSON object."
            )

        canonical = _canonical_json(parsed)

        object.__setattr__(
            self,
            "parameters_json",
            canonical,
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        name: str,
        version: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> "TransformationRecord":
        return cls(
            name=name,
            version=version,
            parameters_json=_canonical_json(
                parameters or {}
            ),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(
            self.parameters_json
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """
    Complete audit metadata for one model feature.

    `raw_inputs` traces the feature back to physical observations.

    `upstream_feature_fingerprints` records intermediate derived
    features when transformations are chained.
    """

    output_name: str
    prediction_time: datetime
    raw_inputs: tuple[RawObservationRef, ...]
    transformation: TransformationRecord
    upstream_feature_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.output_name.strip():
            raise ProvenanceError(
                "output_name cannot be empty."
            )

        if not self.raw_inputs:
            raise ProvenanceError(
                "Feature provenance requires at least one raw input."
            )

        object.__setattr__(
            self,
            "prediction_time",
            ensure_utc(
                self.prediction_time,
                field_name="prediction_time",
            ),
        )

        if self.available_time > self.prediction_time:
            raise ProvenanceError(
                f"Feature {self.output_name!r} uses information "
                f"available after prediction time."
            )

    @property
    def available_time(self) -> datetime:
        return max(
            item.available_time
            for item in self.raw_inputs
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "prediction_time": _iso(
                self.prediction_time
            ),
            "raw_inputs": [
                item.canonical_dict()
                for item in self.raw_inputs
            ],
            "transformation":
                self.transformation.canonical_dict(),
            "upstream_feature_fingerprints":
                list(
                    self.upstream_feature_fingerprints
                ),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()
