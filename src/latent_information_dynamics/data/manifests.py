from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ManifestValidationError(ValueError):
    """Raised when dataset-manifest metadata is invalid."""


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ManifestValidationError(
            "Manifest timestamps must be timezone-aware."
        )

    utc_value = value.astimezone(timezone.utc)

    return utc_value.isoformat().replace("+00:00", "Z")


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 digest of a file without loading the
    entire file into memory.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    normalized = value.lower().strip()

    if len(normalized) != 64:
        raise ManifestValidationError(
            "SHA-256 digest must contain exactly 64 hexadecimal characters."
        )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ManifestValidationError(
            "SHA-256 digest contains non-hexadecimal characters."
        ) from exc

    return normalized


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """
    Provenance for one upstream information source.

    `source_id` should remain stable across dataset builds.
    """

    source_id: str
    provider: str
    dataset: str
    version: str
    retrieval_method: str

    def __post_init__(self) -> None:
        fields = {
            "source_id": self.source_id,
            "provider": self.provider,
            "dataset": self.dataset,
            "version": self.version,
            "retrieval_method": self.retrieval_method,
        }

        for name, value in fields.items():
            if not value.strip():
                raise ManifestValidationError(
                    f"{name} cannot be empty."
                )

    def canonical_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "provider": self.provider,
            "retrieval_method": self.retrieval_method,
            "source_id": self.source_id,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """
    One immutable physical artifact belonging to a dataset snapshot.
    """

    path: str
    sha256: str
    size_bytes: int
    row_count: int | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ManifestValidationError(
                "Artifact path cannot be empty."
            )

        object.__setattr__(
            self,
            "sha256",
            _validate_sha256(self.sha256),
        )

        if self.size_bytes < 0:
            raise ManifestValidationError(
                "size_bytes cannot be negative."
            )

        if self.row_count is not None and self.row_count < 0:
            raise ManifestValidationError(
                "row_count cannot be negative."
            )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
        row_count: int | None = None,
    ) -> "DatasetArtifact":
        file_path = Path(path)

        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        if relative_to is None:
            manifest_path = file_path.as_posix()
        else:
            manifest_path = file_path.relative_to(
                Path(relative_to)
            ).as_posix()

        return cls(
            path=manifest_path,
            sha256=sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
            row_count=row_count,
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """
    Immutable description of one reproducible dataset snapshot.

    The dataset fingerprint intentionally excludes `created_at`.

    Two manifests describing identical source versions, schemas and
    artifact contents should therefore receive the same fingerprint
    even if the manifest files were generated at different times.
    """

    dataset_name: str
    schema_version: str
    created_at: datetime
    sources: tuple[SourceLineage, ...]
    artifacts: tuple[DatasetArtifact, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ManifestValidationError(
                "dataset_name cannot be empty."
            )

        if not self.schema_version.strip():
            raise ManifestValidationError(
                "schema_version cannot be empty."
            )

        if self.created_at.tzinfo is None:
            raise ManifestValidationError(
                "created_at must be timezone-aware."
            )

        if not self.sources:
            raise ManifestValidationError(
                "A manifest must contain at least one source."
            )

        if not self.artifacts:
            raise ManifestValidationError(
                "A manifest must contain at least one artifact."
            )

        source_ids = [
            source.source_id
            for source in self.sources
        ]

        if len(source_ids) != len(set(source_ids)):
            raise ManifestValidationError(
                "source_id values must be unique."
            )

        artifact_paths = [
            artifact.path
            for artifact in self.artifacts
        ]

        if len(artifact_paths) != len(set(artifact_paths)):
            raise ManifestValidationError(
                "Artifact paths must be unique."
            )

    def fingerprint_payload(self) -> dict[str, Any]:
        """
        Canonical content that determines dataset identity.

        Ordering is normalized because source/artifact tuple order
        does not define the underlying data snapshot.
        """
        sources = sorted(
            (
                source.canonical_dict()
                for source in self.sources
            ),
            key=lambda item: item["source_id"],
        )

        artifacts = sorted(
            (
                artifact.canonical_dict()
                for artifact in self.artifacts
            ),
            key=lambda item: item["path"],
        )

        return {
            "artifacts": artifacts,
            "dataset_name": self.dataset_name,
            "schema_version": self.schema_version,
            "sources": sources,
        }

    def canonical_bytes(self) -> bytes:
        payload = self.fingerprint_payload()

        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        return serialized.encode("utf-8")

    @property
    def fingerprint(self) -> str:
        """
        SHA-256 identity of the logical dataset snapshot.
        """
        return hashlib.sha256(
            self.canonical_bytes()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.fingerprint_payload(),
            "created_at": _utc_iso(self.created_at),
            "description": self.description,
            "fingerprint": self.fingerprint,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    def write_json(
        self,
        path: str | Path,
    ) -> None:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            self.to_json() + "\n",
            encoding="utf-8",
        )


def manifest_fingerprint(
    *,
    dataset_name: str,
    schema_version: str,
    sources: Iterable[SourceLineage],
    artifacts: Iterable[DatasetArtifact],
) -> str:
    """
    Convenience function for computing dataset identity without
    permanently writing a manifest.

    The synthetic creation timestamp is excluded from the hash.
    """
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        schema_version=schema_version,
        created_at=datetime.now(timezone.utc),
        sources=tuple(sources),
        artifacts=tuple(artifacts),
    )

    return manifest.fingerprint
