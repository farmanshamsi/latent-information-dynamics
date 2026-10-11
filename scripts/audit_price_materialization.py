from __future__ import annotations

import csv
import json
import math

from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path("data/derived/price_baseline")

FEATURE_FILE = (
    ROOT
    / "btc_price_features_4h_2020_2025.csv"
)

REJECTION_FILE = (
    ROOT
    / "btc_price_features_4h_rejections_2020_2025.csv"
)

METADATA_FILE = (
    ROOT
    / "btc_price_features_4h_2020_2025.json"
)

SOURCE_MANIFEST = Path(
    "data/manifests/binance_btcusdt_5m_2020_2025.json"
)

STEP = timedelta(hours=4)


def main() -> None:
    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    source = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    with FEATURE_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        feature_rows = list(reader)
        feature_header = reader.fieldnames or []

    with REJECTION_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rejection_rows = list(
            csv.DictReader(handle)
        )

    start = datetime.fromisoformat(
        metadata["start_time"]
    )

    end = datetime.fromisoformat(
        metadata["end_time_exclusive"]
    )

    expected_times = []
    cursor = start

    while cursor < end:
        expected_times.append(cursor)
        cursor += STEP

    accepted_times = [
        datetime.fromisoformat(
            row["decision_time"]
        )
        for row in feature_rows
    ]

    rejected_times = [
        datetime.fromisoformat(
            row["decision_time"]
        )
        for row in rejection_rows
    ]

    assert (
        len(expected_times)
        == metadata["theoretical_decisions"]
    )

    assert (
        len(feature_rows)
        == metadata["accepted_decisions"]
    )

    assert (
        len(rejection_rows)
        == metadata["rejected_decisions"]
    )

    assert (
        len(feature_rows)
        + len(rejection_rows)
        == len(expected_times)
    )

    assert (
        feature_header
        == [
            "decision_time",
            *metadata["features"],
        ]
    )

    assert (
        len(set(accepted_times))
        == len(accepted_times)
    )

    assert (
        len(set(rejected_times))
        == len(rejected_times)
    )

    assert not (
        set(accepted_times)
        & set(rejected_times)
    )

    assert (
        set(accepted_times)
        | set(rejected_times)
        == set(expected_times)
    )

    assert accepted_times == sorted(
        accepted_times
    )

    assert rejected_times == sorted(
        rejected_times
    )

    for value in (
        accepted_times
        + rejected_times
    ):
        assert value.tzinfo is not None
        assert value.minute == 0
        assert value.second == 0
        assert value.microsecond == 0
        assert value.hour % 4 == 0

    for row in feature_rows:
        for feature in metadata["features"]:
            value = float(
                row[feature]
            )

            assert math.isfinite(value)

    actual_rejections = {
        "missing_bar": 0,
        "irregular_bar": 0,
        "other": 0,
    }

    for row in rejection_rows:
        reason = row["reason"]

        if reason.startswith("Missing bar"):
            actual_rejections[
                "missing_bar"
            ] += 1

        elif reason.startswith(
            "Irregular bar"
        ):
            actual_rejections[
                "irregular_bar"
            ] += 1

        else:
            actual_rejections[
                "other"
            ] += 1

    actual_rejections = {
        key: value
        for key, value
        in actual_rejections.items()
        if value
    }

    assert (
        actual_rejections
        == metadata["rejection_counts"]
    )

    assert (
        metadata[
            "source_dataset_fingerprint_sha256"
        ]
        == source[
            "dataset_fingerprint_sha256"
        ]
    )

    acceptance_rate = (
        len(feature_rows)
        / len(expected_times)
    )

    print(
        "PRICE MATERIALIZATION AUDIT"
    )
    print(
        "---------------------------"
    )
    print(
        f"Theoretical:       "
        f"{len(expected_times):,}"
    )
    print(
        f"Accepted:          "
        f"{len(feature_rows):,}"
    )
    print(
        f"Rejected:          "
        f"{len(rejection_rows):,}"
    )
    print(
        f"Acceptance rate:   "
        f"{acceptance_rate:.4%}"
    )
    print(
        f"Feature count:     "
        f"{len(metadata["features"])}"
    )

    if accepted_times:
        print(
            f"First accepted:    "
            f"{accepted_times[0].isoformat()}"
        )
        print(
            f"Last accepted:     "
            f"{accepted_times[-1].isoformat()}"
        )

    if rejected_times:
        print(
            f"First rejected:    "
            f"{rejected_times[0].isoformat()}"
        )
        print(
            f"Last rejected:     "
            f"{rejected_times[-1].isoformat()}"
        )

    print(
        f"Rejection causes:  "
        f"{actual_rejections}"
    )

    print()
    print(
        "AUDIT PASSED — every theoretical "
        "decision is accounted for exactly once."
    )


if __name__ == "__main__":
    main()
