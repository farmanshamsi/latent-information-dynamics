from __future__ import annotations

import hashlib
import json
import math
import re

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


class ExperimentSpecError(ValueError):
    """Raised when an experiment specification is invalid."""


_GIT_SHA_RE = re.compile(
    r"^[0-9a-fA-F]{7,64}$"
)


def _utc_iso(value: datetime) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ExperimentSpecError(
            "Experiment timestamps must be timezone-aware."
        )

    return (
        value
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_sha256(
    value: str,
    *,
    field_name: str,
) -> str:
    normalized = value.strip().lower()

    if len(normalized) != 64:
        raise ExperimentSpecError(
            f"{field_name} must contain 64 hexadecimal characters."
        )

    try:
        int(
            normalized,
            16,
        )
    except ValueError as exc:
        raise ExperimentSpecError(
            f"{field_name} must be hexadecimal."
        ) from exc

    return normalized


def _canonical_json_object(
    value: Mapping[str, Any] | None,
) -> str:
    payload = dict(
        value or {}
    )

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ExperimentSpecError(
            "Configuration values must be finite "
            "JSON-serializable objects."
        ) from exc


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """
    Reproducible definition of one model feature.
    """

    name: str
    domain: str
    version: str
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ExperimentSpecError(
                "Feature name cannot be empty."
            )

        if not self.domain.strip():
            raise ExperimentSpecError(
                "Feature domain cannot be empty."
            )

        if not self.version.strip():
            raise ExperimentSpecError(
                "Feature version cannot be empty."
            )

        try:
            parameters = json.loads(
                self.parameters_json
            )
        except json.JSONDecodeError as exc:
            raise ExperimentSpecError(
                "Feature parameters_json must be valid JSON."
            ) from exc

        if not isinstance(
            parameters,
            dict,
        ):
            raise ExperimentSpecError(
                "Feature parameters must be a JSON object."
            )

        canonical = _canonical_json_object(
            parameters
        )

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
        domain: str,
        version: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> "FeatureSpec":
        return cls(
            name=name,
            domain=domain,
            version=version,
            parameters_json=_canonical_json_object(
                parameters
            ),
        )

    @property
    def parameters(
        self,
    ) -> dict[str, Any]:
        return json.loads(
            self.parameters_json
        )

    def canonical_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "parameters": self.parameters,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """
    Definition of the prediction target.
    """

    name: str
    horizon_seconds: int
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ExperimentSpecError(
                "Target name cannot be empty."
            )

        if self.horizon_seconds <= 0:
            raise ExperimentSpecError(
                "horizon_seconds must be positive."
            )

        if not math.isfinite(
            self.threshold
        ):
            raise ExperimentSpecError(
                "Target threshold must be finite."
            )

    def canonical_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "horizon_seconds":
                self.horizon_seconds,
            "name":
                self.name,
            "threshold":
                self.threshold,
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Reproducible model specification.
    """

    family: str
    version: str
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ExperimentSpecError(
                "Model family cannot be empty."
            )

        if not self.version.strip():
            raise ExperimentSpecError(
                "Model version cannot be empty."
            )

        try:
            parameters = json.loads(
                self.parameters_json
            )
        except json.JSONDecodeError as exc:
            raise ExperimentSpecError(
                "Model parameters_json must be valid JSON."
            ) from exc

        if not isinstance(
            parameters,
            dict,
        ):
            raise ExperimentSpecError(
                "Model parameters must be a JSON object."
            )

        object.__setattr__(
            self,
            "parameters_json",
            _canonical_json_object(
                parameters
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        family: str,
        version: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> "ModelSpec":
        return cls(
            family=family,
            version=version,
            parameters_json=_canonical_json_object(
                parameters
            ),
        )

    @property
    def parameters(
        self,
    ) -> dict[str, Any]:
        return json.loads(
            self.parameters_json
        )

    def canonical_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "family": self.family,
            "parameters": self.parameters,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    """
    Chronological evaluation configuration.
    """

    n_folds: int
    min_train_observations: int
    test_observations: int

    purge_observations: int = 0
    embargo_observations: int = 0
    expanding: bool = True

    def __post_init__(self) -> None:
        if self.n_folds <= 0:
            raise ExperimentSpecError(
                "n_folds must be positive."
            )

        if self.min_train_observations <= 0:
            raise ExperimentSpecError(
                "min_train_observations must be positive."
            )

        if self.test_observations <= 0:
            raise ExperimentSpecError(
                "test_observations must be positive."
            )

        if self.purge_observations < 0:
            raise ExperimentSpecError(
                "purge_observations cannot be negative."
            )

        if self.embargo_observations < 0:
            raise ExperimentSpecError(
                "embargo_observations cannot be negative."
            )

    def canonical_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "embargo_observations":
                self.embargo_observations,
            "expanding":
                self.expanding,
            "min_train_observations":
                self.min_train_observations,
            "n_folds":
                self.n_folds,
            "purge_observations":
                self.purge_observations,
            "test_observations":
                self.test_observations,
        }


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """
    Immutable identity of one research experiment.

    `created_at`, `display_name` and `notes` are metadata.

    They deliberately do NOT determine experiment identity.
    """

    display_name: str
    created_at: datetime

    dataset_fingerprint: str
    git_commit: str

    experiment_group: str

    features: tuple[
        FeatureSpec,
        ...
    ]

    target: TargetSpec
    model: ModelSpec
    validation: WalkForwardSpec

    random_seed: int

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise ExperimentSpecError(
                "display_name cannot be empty."
            )

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ExperimentSpecError(
                "created_at must be timezone-aware."
            )

        object.__setattr__(
            self,
            "dataset_fingerprint",
            _validate_sha256(
                self.dataset_fingerprint,
                field_name="dataset_fingerprint",
            ),
        )

        commit = (
            self.git_commit
            .strip()
            .lower()
        )

        if not _GIT_SHA_RE.fullmatch(
            commit
        ):
            raise ExperimentSpecError(
                "git_commit must be a hexadecimal Git "
                "commit identifier."
            )

        object.__setattr__(
            self,
            "git_commit",
            commit,
        )

        if not self.experiment_group.strip():
            raise ExperimentSpecError(
                "experiment_group cannot be empty."
            )

        if not self.features:
            raise ExperimentSpecError(
                "At least one feature is required."
            )

        feature_names = [
            feature.name
            for feature in self.features
        ]

        if len(feature_names) != len(
            set(feature_names)
        ):
            raise ExperimentSpecError(
                "Feature names must be unique "
                "within an experiment."
            )

        if self.random_seed < 0:
            raise ExperimentSpecError(
                "random_seed cannot be negative."
            )

    def identity_payload(
        self,
    ) -> dict[str, Any]:
        """
        Content determining scientific experiment identity.

        Feature order is retained deliberately because model input
        ordering may be meaningful.
        """
        return {
            "dataset_fingerprint":
                self.dataset_fingerprint,
            "experiment_group":
                self.experiment_group,
            "features": [
                feature.canonical_dict()
                for feature in self.features
            ],
            "git_commit":
                self.git_commit,
            "model":
                self.model.canonical_dict(),
            "random_seed":
                self.random_seed,
            "target":
                self.target.canonical_dict(),
            "validation":
                self.validation.canonical_dict(),
        }

    def canonical_bytes(
        self,
    ) -> bytes:
        return json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode(
            "utf-8"
        )

    @property
    def fingerprint(
        self,
    ) -> str:
        return hashlib.sha256(
            self.canonical_bytes()
        ).hexdigest()

    @property
    def experiment_id(
        self,
    ) -> str:
        return (
            "exp-"
            + self.fingerprint[:20]
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "created_at":
                _utc_iso(
                    self.created_at
                ),
            "display_name":
                self.display_name,
            "experiment_id":
                self.experiment_id,
            "fingerprint":
                self.fingerprint,
            "notes":
                self.notes,
        }

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
