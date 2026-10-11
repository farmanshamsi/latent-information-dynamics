from __future__ import annotations

import json
import re

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .spec import (
    ExperimentSpec,
    ExperimentSpecError,
)


class ExperimentRegistryError(RuntimeError):
    """Raised when the experiment registry cannot preserve immutability."""


_RUN_ID_RE = re.compile(
    r"^[A-Za-z0-9._-]+$"
)


def _utc_iso(
    value: datetime,
) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ExperimentRegistryError(
            "Result timestamps must be timezone-aware."
        )

    return (
        value
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json_object(
    value: Mapping[str, Any],
) -> str:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ExperimentRegistryError(
            "Result metrics must be finite "
            "JSON-serializable values."
        ) from exc


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """
    Immutable result record attached to a registered experiment.

    A single experiment may have several run IDs, for example when
    executing the same frozen specification on different machines.
    """

    experiment_id: str
    run_id: str
    created_at: datetime
    metrics_json: str
    artifacts: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ExperimentRegistryError(
                "experiment_id cannot be empty."
            )

        if not _RUN_ID_RE.fullmatch(
            self.run_id
        ):
            raise ExperimentRegistryError(
                "run_id may contain only letters, numbers, "
                "periods, underscores and hyphens."
            )

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ExperimentRegistryError(
                "created_at must be timezone-aware."
            )

        try:
            metrics = json.loads(
                self.metrics_json
            )
        except json.JSONDecodeError as exc:
            raise ExperimentRegistryError(
                "metrics_json must be valid JSON."
            ) from exc

        if not isinstance(
            metrics,
            dict,
        ):
            raise ExperimentRegistryError(
                "Result metrics must be a JSON object."
            )

        object.__setattr__(
            self,
            "metrics_json",
            _canonical_json_object(
                metrics
            ),
        )

        for artifact in self.artifacts:
            if not artifact.strip():
                raise ExperimentRegistryError(
                    "Artifact paths cannot be empty."
                )

    @classmethod
    def from_mapping(
        cls,
        *,
        experiment_id: str,
        run_id: str,
        created_at: datetime,
        metrics: Mapping[str, Any],
        artifacts: tuple[str, ...] = (),
        notes: str = "",
    ) -> "ExperimentResult":
        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            created_at=created_at,
            metrics_json=_canonical_json_object(
                metrics
            ),
            artifacts=artifacts,
            notes=notes,
        )

    @property
    def metrics(
        self,
    ) -> dict[str, Any]:
        return json.loads(
            self.metrics_json
        )

    def identity_payload(
        self,
    ) -> dict[str, Any]:
        return {
            "artifacts":
                list(
                    self.artifacts
                ),
            "experiment_id":
                self.experiment_id,
            "metrics":
                self.metrics,
            "run_id":
                self.run_id,
        }

    def fingerprint(
        self,
    ) -> str:
        import hashlib

        payload = json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode(
            "utf-8"
        )

        return hashlib.sha256(
            payload
        ).hexdigest()

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "created_at":
                _utc_iso(
                    self.created_at
                ),
            "fingerprint":
                self.fingerprint(),
            "notes":
                self.notes,
        }


class ExperimentRegistry:
    """
    Filesystem-backed immutable experiment registry.

    Directory layout:

    registry/
        exp-xxxxxxxx/
            spec.json
            results/
                run-001.json
    """

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(
            root
        )

    def _experiment_dir(
        self,
        experiment_id: str,
    ) -> Path:
        return (
            self.root
            / experiment_id
        )

    def _spec_path(
        self,
        experiment_id: str,
    ) -> Path:
        return (
            self._experiment_dir(
                experiment_id
            )
            / "spec.json"
        )

    def _result_path(
        self,
        *,
        experiment_id: str,
        run_id: str,
    ) -> Path:
        return (
            self._experiment_dir(
                experiment_id
            )
            / "results"
            / f"{run_id}.json"
        )

    def register(
        self,
        spec: ExperimentSpec,
    ) -> Path:
        experiment_dir = (
            self._experiment_dir(
                spec.experiment_id
            )
        )

        spec_path = (
            experiment_dir
            / "spec.json"
        )

        if spec_path.exists():
            existing = json.loads(
                spec_path.read_text(
                    encoding="utf-8"
                )
            )

            existing_fingerprint = existing.get(
                "fingerprint"
            )

            if (
                existing_fingerprint
                != spec.fingerprint
            ):
                raise ExperimentRegistryError(
                    "Experiment ID collision or attempted "
                    "mutation of an existing experiment."
                )

            return spec_path

        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            experiment_dir
            / "results"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            experiment_dir
            / "spec.json.tmp"
        )

        temporary.write_text(
            spec.to_json() + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            spec_path
        )

        return spec_path

    def exists(
        self,
        experiment_id: str,
    ) -> bool:
        return self._spec_path(
            experiment_id
        ).is_file()

    def load(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        path = self._spec_path(
            experiment_id
        )

        if not path.is_file():
            raise ExperimentRegistryError(
                f"Unknown experiment {experiment_id!r}."
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    def list_ids(
        self,
    ) -> tuple[str, ...]:
        if not self.root.exists():
            return ()

        experiment_ids = []

        for path in self.root.iterdir():
            if (
                path.is_dir()
                and (
                    path
                    / "spec.json"
                ).is_file()
            ):
                experiment_ids.append(
                    path.name
                )

        return tuple(
            sorted(
                experiment_ids
            )
        )

    def register_result(
        self,
        result: ExperimentResult,
    ) -> Path:
        if not self.exists(
            result.experiment_id
        ):
            raise ExperimentRegistryError(
                f"Cannot attach result to unknown experiment "
                f"{result.experiment_id!r}."
            )

        result_path = self._result_path(
            experiment_id=
                result.experiment_id,
            run_id=
                result.run_id,
        )

        if result_path.exists():
            existing = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                existing.get(
                    "fingerprint"
                )
                != result.fingerprint()
            ):
                raise ExperimentRegistryError(
                    f"Run ID {result.run_id!r} already exists "
                    f"with different results."
                )

            return result_path

        result_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = result_path.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(
                result.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            result_path
        )

        return result_path
