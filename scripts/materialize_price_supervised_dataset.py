from __future__ import annotations

import csv
import hashlib
import json
import math

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from latent_information_dynamics.data.feature_frame import (
    FeatureFrameRow,
)
from latent_information_dynamics.datasets.supervised import (
    build_supervised_dataset,
)
from latent_information_dynamics.targets.binary import (
    DecisionWindow,
    ReferencePriceObservation,
    TieRule,
    build_binary_direction_target,
)


FEATURE_FILE = Path(
    "data/derived/price_baseline/"
    "btc_price_features_4h_2020_2025.csv"
)

FEATURE_METADATA_FILE = Path(
    "data/derived/price_baseline/"
    "btc_price_features_4h_2020_2025.json"
)

TARGET_FILE = Path(
    "data/derived/targets/"
    "btc_binary_targets_4h_2020_2025.csv"
)

TARGET_REJECTION_FILE = Path(
    "data/derived/targets/"
    "btc_binary_targets_4h_rejections_2020_2025.csv"
)

TARGET_METADATA_FILE = Path(
    "data/derived/targets/"
    "btc_binary_targets_4h_2020_2025.json"
)

OUTPUT_ROOT = Path(
    "data/derived/supervised"
)

OUTPUT_FILE = (
    OUTPUT_ROOT
    / "btc_price_supervised_4h_2020_2025.csv"
)

METADATA_FILE = (
    OUTPUT_ROOT
    / "btc_price_supervised_4h_2020_2025.json"
)

FOUR_HOURS = timedelta(hours=4)


@dataclass(frozen=True, slots=True)
class FrozenNumericFeature:
    name: str
    value: float
    available_time: datetime


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    print(
        "Loading frozen price features..."
    )

    feature_metadata = json.loads(
        FEATURE_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    target_metadata = json.loads(
        TARGET_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    with FEATURE_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Feature CSV has no header."
            )

        feature_names = tuple(
            name
            for name in reader.fieldnames
            if name != "decision_time"
        )

        feature_rows = {}

        for row in reader:
            decision_time = (
                datetime.fromisoformat(
                    row["decision_time"]
                )
            )

            if decision_time in feature_rows:
                raise RuntimeError(
                    "Duplicate feature decision time."
                )

            values = tuple(
                float(row[name])
                for name in feature_names
            )

            if not all(
                math.isfinite(value)
                for value in values
            ):
                raise RuntimeError(
                    "Non-finite frozen feature."
                )

            features = tuple(
                FrozenNumericFeature(
                    name=name,
                    value=value,
                    available_time=decision_time,
                )
                for name, value
                in zip(
                    feature_names,
                    values,
                    strict=True,
                )
            )

            feature_rows[
                decision_time
            ] = FeatureFrameRow(
                prediction_time=decision_time,
                features=features,
            )

    print(
        "Loading frozen binary targets..."
    )

    target_rows = {}

    with TARGET_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            decision_time = (
                datetime.fromisoformat(
                    row["decision_time"]
                )
            )

            resolution_time = (
                datetime.fromisoformat(
                    row["resolution_time"]
                )
            )

            if decision_time in target_rows:
                raise RuntimeError(
                    "Duplicate target decision time."
                )

            start_reference = (
                ReferencePriceObservation(
                    source=row[
                        "reference_source"
                    ],
                    reference_time=
                        decision_time,
                    available_time=
                        datetime.fromisoformat(
                            row[
                                "start_available_time"
                            ]
                        ),
                    price=float(
                        row["start_price"]
                    ),
                )
            )

            end_reference = (
                ReferencePriceObservation(
                    source=row[
                        "reference_source"
                    ],
                    reference_time=
                        resolution_time,
                    available_time=
                        datetime.fromisoformat(
                            row[
                                "end_available_time"
                            ]
                        ),
                    price=float(
                        row["end_price"]
                    ),
                )
            )

            target = (
                build_binary_direction_target(
                    window=DecisionWindow(
                        decision_time=
                            decision_time,
                        resolution_time=
                            resolution_time,
                    ),
                    start_reference=
                        start_reference,
                    end_reference=
                        end_reference,
                    tie_rule=TieRule(
                        row["tie_rule"]
                    ),
                )
            )

            expected_outcome = int(
                row["outcome"]
            )

            if (
                target.outcome
                != expected_outcome
            ):
                raise RuntimeError(
                    "Reconstructed target "
                    "outcome mismatch."
                )

            expected_return = float(
                row["arithmetic_return"]
            )

            expected_log_return = float(
                row["log_return"]
            )

            if not math.isclose(
                target.arithmetic_return,
                expected_return,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise RuntimeError(
                    "Arithmetic return "
                    "reconstruction mismatch."
                )

            if not math.isclose(
                target.log_return,
                expected_log_return,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise RuntimeError(
                    "Log return "
                    "reconstruction mismatch."
                )

            target_rows[
                decision_time
            ] = target

    with TARGET_REJECTION_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rejected_target_times = {
            datetime.fromisoformat(
                row["decision_time"]
            )
            for row in csv.DictReader(
                handle
            )
        }

    feature_times = set(
        feature_rows
    )

    target_times = set(
        target_rows
    )

    feature_without_target = (
        feature_times
        - target_times
    )

    target_without_feature = (
        target_times
        - feature_times
    )

    if target_without_feature:
        raise RuntimeError(
            "A target exists without "
            "a valid feature row."
        )

    if (
        feature_without_target
        != rejected_target_times
    ):
        raise RuntimeError(
            "Feature-target mismatch does "
            "not exactly match recorded "
            "target rejections."
        )

    joined_times = sorted(
        feature_times
        & target_times
    )

    pairs = [
        (
            feature_rows[
                decision_time
            ],
            target_rows[
                decision_time
            ],
        )
        for decision_time
        in joined_times
    ]

    print(
        "Building supervised dataset "
        "through existing PIT safeguards..."
    )

    dataset = build_supervised_dataset(
        pairs
    )

    matrix = dataset.matrix()

    if len(dataset.examples) != 13030:
        raise RuntimeError(
            "Expected exactly 13,030 "
            "supervised examples."
        )

    if len(matrix.X) != 13030:
        raise RuntimeError(
            "Matrix row count mismatch."
        )

    if len(matrix.y) != 13030:
        raise RuntimeError(
            "Target vector count mismatch."
        )

    if (
        matrix.feature_names
        != feature_names
    ):
        raise RuntimeError(
            "Feature schema changed "
            "during supervised join."
        )

    if len(feature_names) != 11:
        raise RuntimeError(
            "Expected exactly 11 "
            "price features."
        )

    up = 0
    down = 0

    for example in dataset.examples:
        if (
            example.resolution_time
            != example.decision_time
            + FOUR_HOURS
        ):
            raise RuntimeError(
                "Unexpected target horizon."
            )

        if (
            example.label_available_time
            != example.resolution_time
        ):
            raise RuntimeError(
                "Label availability must "
                "equal resolution time for "
                "this historical reference."
            )

        if (
            example.label_available_time
            <= example.decision_time
        ):
            raise RuntimeError(
                "Target leakage detected."
            )

        if example.target == 1:
            up += 1
        elif example.target == 0:
            down += 1
        else:
            raise RuntimeError(
                "Invalid binary target."
            )

        if not all(
            math.isfinite(value)
            for value
            in example.feature_values
        ):
            raise RuntimeError(
                "Non-finite model feature."
            )

    if (
        up
        != target_metadata[
            "class_balance"
        ]["up"]
    ):
        raise RuntimeError(
            "UP count differs from "
            "target metadata."
        )

    if (
        down
        != target_metadata[
            "class_balance"
        ]["down"]
    ):
        raise RuntimeError(
            "DOWN count differs from "
            "target metadata."
        )

    if (
        feature_names
        != tuple(
            feature_metadata[
                "features"
            ]
        )
    ):
        raise RuntimeError(
            "Feature schema differs from "
            "price metadata."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_time",
                "resolution_time",
                "label_available_time",
                *feature_names,
                "target",
                "realized_return",
                "realized_log_return",
            ],
        )

        writer.writeheader()

        for example in dataset.examples:
            row = {
                "decision_time":
                    example.decision_time.isoformat(),
                "resolution_time":
                    example.resolution_time.isoformat(),
                "label_available_time":
                    example.label_available_time.isoformat(),
                "target":
                    example.target,
                "realized_return":
                    example.realized_return,
                "realized_log_return":
                    example.realized_log_return,
            }

            row.update(
                example.feature_dict()
            )

            writer.writerow(row)

    output_metadata = {
        "dataset":
            "btc-price-supervised-4h-2020-2025",
        "rows":
            len(dataset.examples),
        "feature_count":
            len(feature_names),
        "features":
            list(feature_names),
        "target_definition":
            target_metadata[
                "label_definition"
            ],
        "horizon_hours":
            4,
        "up":
            up,
        "down":
            down,
        "up_share":
            up / (
                up + down
            ),
        "feature_valid_decisions":
            len(feature_rows),
        "target_valid_decisions":
            len(target_rows),
        "final_joined_decisions":
            len(dataset.examples),
        "feature_rows_without_target":
            len(
                feature_without_target
            ),
        "target_rows_without_feature":
            len(
                target_without_feature
            ),
        "first_decision_time":
            dataset.examples[
                0
            ].decision_time.isoformat(),
        "last_decision_time":
            dataset.examples[
                -1
            ].decision_time.isoformat(),
        "label_availability_rule":
            (
                "A label becomes available "
                "only at the future "
                "resolution boundary."
            ),
        "feature_availability_rule":
            (
                "Frozen price features are "
                "treated as available at "
                "their decision boundary "
                "because each was previously "
                "materialized exclusively "
                "from completed bars."
            ),
        "join_policy":
            (
                "Exact decision_time "
                "intersection only. No "
                "nearest-time matching, "
                "imputation, interpolation "
                "or forward filling."
            ),
        "source_price_feature_sha256":
            sha256_file(
                FEATURE_FILE
            ),
        "source_target_sha256":
            sha256_file(
                TARGET_FILE
            ),
        "source_target_rejections_sha256":
            sha256_file(
                TARGET_REJECTION_FILE
            ),
        "source_raw_dataset_fingerprint_sha256":
            target_metadata[
                "source_raw_dataset_fingerprint_sha256"
            ],
    }

    METADATA_FILE.write_text(
        json.dumps(
            output_metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "SUPERVISED DATASET AUDIT"
    )
    print(
        "------------------------"
    )
    print(
        f"Feature rows:        "
        f"{len(feature_rows):,}"
    )
    print(
        f"Target rows:         "
        f"{len(target_rows):,}"
    )
    print(
        f"Feature-only rows:   "
        f"{len(feature_without_target):,}"
    )
    print(
        f"Target-only rows:    "
        f"{len(target_without_feature):,}"
    )
    print(
        f"Final examples:      "
        f"{len(dataset.examples):,}"
    )
    print(
        f"Feature count:       "
        f"{len(feature_names)}"
    )
    print(
        f"UP:                  "
        f"{up:,}"
    )
    print(
        f"DOWN:                "
        f"{down:,}"
    )
    print(
        f"UP share:            "
        f"{up / (up + down):.4%}"
    )
    print(
        f"First decision:      "
        f"{dataset.examples[0].decision_time.isoformat()}"
    )
    print(
        f"Last decision:       "
        f"{dataset.examples[-1].decision_time.isoformat()}"
    )
    print()
    print(
        "LEAKAGE AUDIT PASSED"
    )
    print(
        "Every feature is available "
        "by decision time."
    )
    print(
        "Every label becomes available "
        "only at resolution time."
    )
    print(
        "Every unmatched feature row "
        "corresponds exactly to a "
        "recorded target rejection."
    )
    print()
    print(
        f"Dataset  -> {OUTPUT_FILE}"
    )
    print(
        f"Metadata -> {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()
