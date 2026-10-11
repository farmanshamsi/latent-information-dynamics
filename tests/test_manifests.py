from datetime import datetime, timezone

import json

import pytest

from latent_information_dynamics.data.manifests import (
    DatasetArtifact,
    DatasetManifest,
    ManifestValidationError,
    SourceLineage,
    sha256_file,
)


UTC = timezone.utc


def source(
    *,
    source_id: str = "btc-market",
    version: str = "v1",
) -> SourceLineage:
    return SourceLineage(
        source_id=source_id,
        provider="example-provider",
        dataset="BTC market data",
        version=version,
        retrieval_method="historical-api",
    )


def artifact(
    *,
    path: str = "btc.parquet",
    digest: str = "a" * 64,
) -> DatasetArtifact:
    return DatasetArtifact(
        path=path,
        sha256=digest,
        size_bytes=100,
        row_count=10,
    )


def manifest(
    *,
    created_at: datetime,
    sources: tuple[SourceLineage, ...] | None = None,
    artifacts: tuple[DatasetArtifact, ...] | None = None,
) -> DatasetManifest:
    return DatasetManifest(
        dataset_name="bitcoin-research",
        schema_version="1.0.0",
        created_at=created_at,
        sources=sources or (source(),),
        artifacts=artifacts or (artifact(),),
        description="test manifest",
    )


def test_sha256_file_is_deterministic(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"bitcoin")

    digest_1 = sha256_file(path)
    digest_2 = sha256_file(path)

    assert digest_1 == digest_2
    assert len(digest_1) == 64


def test_file_content_change_changes_digest(tmp_path) -> None:
    path = tmp_path / "sample.txt"

    path.write_bytes(b"bitcoin")
    digest_1 = sha256_file(path)

    path.write_bytes(b"bitcoin!")
    digest_2 = sha256_file(path)

    assert digest_1 != digest_2


def test_created_at_does_not_change_dataset_identity() -> None:
    first = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    second = manifest(
        created_at=datetime(
            2026,
            2,
            1,
            tzinfo=UTC,
        )
    )

    assert first.fingerprint == second.fingerprint


def test_artifact_change_changes_fingerprint() -> None:
    first = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    second = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        artifacts=(
            artifact(
                digest="b" * 64,
            ),
        ),
    )

    assert first.fingerprint != second.fingerprint


def test_source_version_change_changes_fingerprint() -> None:
    first = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    second = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        sources=(
            source(
                version="v2",
            ),
        ),
    )

    assert first.fingerprint != second.fingerprint


def test_source_order_does_not_change_fingerprint() -> None:
    source_a = source(
        source_id="a",
    )

    source_b = source(
        source_id="b",
    )

    created_at = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    first = manifest(
        created_at=created_at,
        sources=(
            source_a,
            source_b,
        ),
    )

    second = manifest(
        created_at=created_at,
        sources=(
            source_b,
            source_a,
        ),
    )

    assert first.fingerprint == second.fingerprint


def test_artifact_order_does_not_change_fingerprint() -> None:
    artifact_a = artifact(
        path="a.parquet",
        digest="a" * 64,
    )

    artifact_b = artifact(
        path="b.parquet",
        digest="b" * 64,
    )

    created_at = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    first = manifest(
        created_at=created_at,
        artifacts=(
            artifact_a,
            artifact_b,
        ),
    )

    second = manifest(
        created_at=created_at,
        artifacts=(
            artifact_b,
            artifact_a,
        ),
    )

    assert first.fingerprint == second.fingerprint


def test_invalid_sha256_is_rejected() -> None:
    with pytest.raises(
        ManifestValidationError
    ):
        DatasetArtifact(
            path="bad.parquet",
            sha256="not-a-real-digest",
            size_bytes=10,
        )


def test_duplicate_source_ids_are_rejected() -> None:
    duplicated = (
        source(
            source_id="same",
        ),
        source(
            source_id="same",
            version="v2",
        ),
    )

    with pytest.raises(
        ManifestValidationError
    ):
        manifest(
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ),
            sources=duplicated,
        )


def test_duplicate_artifact_paths_are_rejected() -> None:
    duplicated = (
        artifact(
            path="same.parquet",
            digest="a" * 64,
        ),
        artifact(
            path="same.parquet",
            digest="b" * 64,
        ),
    )

    with pytest.raises(
        ManifestValidationError
    ):
        manifest(
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ),
            artifacts=duplicated,
        )


def test_manifest_json_contains_fingerprint(
    tmp_path,
) -> None:
    item = manifest(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    output = tmp_path / "manifest.json"

    item.write_json(output)

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert payload["fingerprint"] == item.fingerprint
    assert payload["schema_version"] == "1.0.0"


def test_artifact_from_file_records_content(
    tmp_path,
) -> None:
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"abc123")

    item = DatasetArtifact.from_file(
        path,
        relative_to=tmp_path,
        row_count=1,
    )

    assert item.path == "dataset.bin"
    assert item.size_bytes == 6
    assert item.row_count == 1
    assert item.sha256 == sha256_file(path)
