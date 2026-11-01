from __future__ import annotations

import csv
import hashlib
import json

from datetime import datetime, timedelta
from pathlib import Path

from latent_information_dynamics.datasets.target_materialization import (
    materialize_binary_targets,
)
from latent_information_dynamics.market_data.binance_csv import (
    read_klines_directory,
)
from latent_information_dynamics.targets.binary import (
    TieRule,
)


RAW_ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

FEATURE_CARRIER_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/supervised"
)

OUTPUT_FILE = (
    OUTPUT_ROOT
    / "btc_combined_supervised_24h_2020_2025.csv"
)

METADATA_FILE = (
    OUTPUT_ROOT
    / "btc_combined_supervised_24h_2020_2025.json"
)

REJECTION_FILE = (
    OUTPUT_ROOT
    / "btc_combined_supervised_24h_rejections_2020_2025.csv"
)

HORIZON = timedelta(
    hours=24
)

TIE_RULE = TieRule.UP

NON_FEATURE_COLUMNS = {
    "decision_time",
    "resolution_time",
    "label_available_time",
    "target",
    "realized_return",
    "realized_log_return",
}


def parse_datetime(
    value: str,
) -> datetime:
    return datetime.fromisoformat(
        value
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    if not FEATURE_CARRIER_FILE.exists():
        raise FileNotFoundError(
            FEATURE_CARRIER_FILE
        )

    print(
        "Loading frozen combined "
        "Price+Flow feature carrier..."
    )

    with FEATURE_CARRIER_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
        )

        if fieldnames is None:
            raise RuntimeError(
                "Missing CSV header."
            )

        source_rows = list(
            reader
        )

    if len(source_rows) != 13030:
        raise RuntimeError(
            "Expected 13,030 frozen "
            "feature decisions, found "
            f"{len(source_rows):,}."
        )

    feature_names = [
        name
        for name in fieldnames
        if name
        not in NON_FEATURE_COLUMNS
    ]

    if len(feature_names) != 31:
        raise RuntimeError(
            "Expected exactly 31 frozen "
            "Price+Flow features, found "
            f"{len(feature_names)}."
        )

    decision_times = [
        parse_datetime(
            row[
                "decision_time"
            ]
        )
        for row in source_rows
    ]

    if decision_times != sorted(
        decision_times
    ):
        raise RuntimeError(
            "Feature carrier is not "
            "chronologically ordered."
        )

    if len(
        set(
            decision_times
        )
    ) != len(
        decision_times
    ):
        raise RuntimeError(
            "Duplicate decision times."
        )

    print(
        "Frozen feature decisions:",
        f"{len(decision_times):,}",
    )

    print(
        "Loading frozen BTC bars..."
    )

    bars = read_klines_directory(
        RAW_ROOT
    )

    print(
        "Loaded bars:",
        f"{len(bars):,}",
    )

    print(
        "Materializing 24-hour "
        "binary targets..."
    )

    result = materialize_binary_targets(
        bars,
        decision_times=
            decision_times,
        horizon=HORIZON,
        tie_rule=TIE_RULE,
    )

    if (
        result.accepted_count
        + result.rejected_count
        != result.requested_decisions
    ):
        raise RuntimeError(
            "Target accounting failure."
        )

    target_by_time = {
        target.window.decision_time:
            target
        for target in result.targets
    }

    if len(
        target_by_time
    ) != result.accepted_count:
        raise RuntimeError(
            "Duplicate accepted target."
        )

    output_rows = []

    up = 0
    down = 0

    for source_row in source_rows:
        decision_time = (
            parse_datetime(
                source_row[
                    "decision_time"
                ]
            )
        )

        target = target_by_time.get(
            decision_time
        )

        if target is None:
            continue

        expected_resolution = (
            decision_time
            + HORIZON
        )

        if (
            target.window
            .resolution_time
            != expected_resolution
        ):
            raise RuntimeError(
                "Unexpected target "
                "resolution."
            )

        row = {
            name:
                source_row[
                    name
                ]
            for name in feature_names
        }

        row.update(
            {
                "decision_time":
                    decision_time
                    .isoformat(),
                "resolution_time":
                    target
                    .window
                    .resolution_time
                    .isoformat(),
                "label_available_time":
                    target
                    .window
                    .resolution_time
                    .isoformat(),
                "target":
                    target.outcome,
                "realized_return":
                    target
                    .arithmetic_return,
                "realized_log_return":
                    target
                    .log_return,
            }
        )

        if target.outcome == 1:
            up += 1
        elif target.outcome == 0:
            down += 1
        else:
            raise RuntimeError(
                "Invalid binary outcome."
            )

        output_rows.append(
            row
        )

    if len(
        output_rows
    ) != result.accepted_count:
        raise RuntimeError(
            "Output/target count mismatch."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_fieldnames = [
        "decision_time",
        "resolution_time",
        "label_available_time",
        *feature_names,
        "target",
        "realized_return",
        "realized_log_return",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                output_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    with REJECTION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_time",
                "resolution_time",
                "reason",
            ],
        )

        writer.writeheader()

        for rejected in (
            result.rejected
        ):
            writer.writerow(
                {
                    "decision_time":
                        rejected
                        .decision_time
                        .isoformat(),
                    "resolution_time":
                        (
                            rejected
                            .decision_time
                            + HORIZON
                        ).isoformat(),
                    "reason":
                        rejected.reason,
                }
            )

    first_time = (
        parse_datetime(
            output_rows[
                0
            ][
                "decision_time"
            ]
        )
    )

    last_time = (
        parse_datetime(
            output_rows[
                -1
            ][
                "decision_time"
            ]
        )
    )

    metadata = {
        "dataset":
            (
                "btc-combined-supervised-"
                "24h-2020-2025"
            ),
        "purpose":
            (
                "Frozen-horizon robustness "
                "dataset."
            ),
        "feature_source":
            str(
                FEATURE_CARRIER_FILE
            ),
        "feature_source_sha256":
            sha256_file(
                FEATURE_CARRIER_FILE
            ),
        "feature_reuse_rule":
            (
                "Price and Flow features are "
                "copied unchanged from the "
                "validated 4h combined feature "
                "carrier. All original future "
                "label fields are discarded and "
                "rematerialized at 24h."
            ),
        "horizon_hours":
            24,
        "tie_rule":
            TIE_RULE.value,
        "requested_decisions":
            result
            .requested_decisions,
        "accepted_targets":
            result
            .accepted_count,
        "rejected_targets":
            result
            .rejected_count,
        "rows":
            len(
                output_rows
            ),
        "feature_count":
            len(
                feature_names
            ),
        "features":
            feature_names,
        "up":
            up,
        "down":
            down,
        "up_share":
            (
                up
                / (
                    up + down
                )
            ),
        "first_decision_time":
            first_time
            .isoformat(),
        "last_decision_time":
            last_time
            .isoformat(),
        "last_resolution_time":
            output_rows[
                -1
            ][
                "resolution_time"
            ],
        "strict_point_in_time":
            True,
        "label_availability_rule":
            (
                "label_available_time equals "
                "24-hour resolution_time."
            ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "24H ROBUSTNESS DATASET"
    )

    print(
        "======================"
    )

    print(
        "Requested:",
        f"{result.requested_decisions:,}",
    )

    print(
        "Accepted:",
        f"{result.accepted_count:,}",
    )

    print(
        "Rejected:",
        f"{result.rejected_count:,}",
    )

    print(
        "Features:",
        len(
            feature_names
        ),
    )

    print(
        "UP:",
        f"{up:,}",
    )

    print(
        "DOWN:",
        f"{down:,}",
    )

    print(
        "UP share:",
        f"{up / (up + down):.4%}",
    )

    print(
        "First decision:",
        first_time.isoformat(),
    )

    print(
        "Last decision:",
        last_time.isoformat(),
    )

    print(
        "Last resolution:",
        output_rows[
            -1
        ][
            "resolution_time"
        ],
    )

    print()
    print(
        "Dataset    ->",
        OUTPUT_FILE,
    )

    print(
        "Metadata   ->",
        METADATA_FILE,
    )

    print(
        "Rejections ->",
        REJECTION_FILE,
    )


if __name__ == "__main__":
    main()
