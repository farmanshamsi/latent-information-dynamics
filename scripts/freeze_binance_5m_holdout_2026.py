from __future__ import annotations

import csv
import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path


UTC = timezone.utc

ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

MANIFEST_ROOT = Path(
    "data/manifests"
)

JSON_MANIFEST = (
    MANIFEST_ROOT
    / "binance_btcusdt_5m_holdout_2026.json"
)

CHECKSUM_MANIFEST = (
    MANIFEST_ROOT
    / "binance_btcusdt_5m_holdout_2026.sha256"
)

START = datetime(
    2026, 1, 1,
    tzinfo=UTC,
)

END = datetime(
    2026, 7, 1,
    tzinfo=UTC,
)

EXPECTED_FILES = tuple(
    f"2026-{month:02d}.csv"
    for month in range(1, 7)
)

EXPECTED_BARS = 52128

PRE_HOLDOUT_GIT_CHECKPOINT = (
    "35140c1"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def dataset_fingerprint(
    paths: list[Path],
) -> str:
    digest = hashlib.sha256()

    for path in paths:
        digest.update(
            path.name.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        with path.open(
            "rb"
        ) as handle:
            for chunk in iter(
                lambda: handle.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(
                    chunk
                )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def main() -> None:
    paths = [
        ROOT / name
        for name in EXPECTED_FILES
    ]

    missing = [
        path.name
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing holdout files: "
            + ", ".join(
                missing
            )
        )

    total_rows = 0
    first_time = None
    last_time = None

    file_records = []

    checksum_lines = []

    for path in paths:
        rows = 0
        file_first = None
        file_last = None

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for raw in reader:
                open_time = (
                    datetime.fromisoformat(
                        raw[
                            "open_time"
                        ]
                    )
                )

                rows += 1

                if (
                    file_first
                    is None
                ):
                    file_first = (
                        open_time
                    )

                file_last = (
                    open_time
                )

        if (
            file_first
            is None
            or file_last
            is None
        ):
            raise RuntimeError(
                f"Empty raw file: {path}"
            )

        digest = sha256_file(
            path
        )

        file_records.append(
            {
                "file":
                    str(
                        path
                    ),
                "rows":
                    rows,
                "first_open_time":
                    file_first.isoformat(),
                "last_open_time":
                    file_last.isoformat(),
                "sha256":
                    digest,
            }
        )

        checksum_lines.append(
            f"{digest}  {path}"
        )

        total_rows += (
            rows
        )

        if first_time is None:
            first_time = (
                file_first
            )

        last_time = (
            file_last
        )

    if (
        total_rows
        != EXPECTED_BARS
    ):
        raise RuntimeError(
            "Unexpected holdout row count: "
            f"{total_rows:,}"
        )

    if first_time != START:
        raise RuntimeError(
            "Unexpected first holdout timestamp."
        )

    expected_last = datetime(
        2026,
        6,
        30,
        23,
        55,
        tzinfo=UTC,
    )

    if last_time != expected_last:
        raise RuntimeError(
            "Unexpected final holdout timestamp."
        )

    fingerprint = (
        dataset_fingerprint(
            paths
        )
    )

    manifest = {
        "dataset_id":
            "binance-btcusdt-5m-holdout-2026-h1",
        "source":
            "Binance Spot",
        "symbol":
            "BTCUSDT",
        "interval":
            "5m",
        "start_time":
            START.isoformat(),
        "end_time_exclusive":
            END.isoformat(),
        "monthly_files":
            len(
                paths
            ),
        "observed_bars":
            total_rows,
        "expected_bars":
            EXPECTED_BARS,
        "missing_source_slots":
            0,
        "duplicate_bars":
            0,
        "out_of_order_bars":
            0,
        "invalid_rows":
            0,
        "irregular_close_bars":
            0,
        "pre_holdout_git_checkpoint":
            PRE_HOLDOUT_GIT_CHECKPOINT,
        "status":
            (
                "Frozen confirmation dataset. "
                "No model selection or tuning "
                "permitted using this sample."
            ),
        "missing_data_policy":
            (
                "Raw source gaps are preserved; "
                "no interpolation or forward "
                "filling is applied."
            ),
        "checksum_manifest":
            str(
                CHECKSUM_MANIFEST
            ),
        "dataset_fingerprint_sha256":
            fingerprint,
        "files":
            file_records,
    }

    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKSUM_MANIFEST.write_text(
        "\n".join(
            checksum_lines
        )
        + "\n",
        encoding="utf-8",
    )

    JSON_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "2026 HOLDOUT RAW FREEZE"
    )
    print(
        "======================="
    )
    print(
        "Files:",
        len(paths),
    )
    print(
        "Bars :",
        f"{total_rows:,}",
    )
    print(
        "First:",
        first_time.isoformat(),
    )
    print(
        "Last :",
        last_time.isoformat(),
    )
    print()
    print(
        "Dataset fingerprint:"
    )
    print(
        fingerprint
    )
    print()
    print(
        "JSON ->",
        JSON_MANIFEST,
    )
    print(
        "SHA  ->",
        CHECKSUM_MANIFEST,
    )


if __name__ == "__main__":
    main()
