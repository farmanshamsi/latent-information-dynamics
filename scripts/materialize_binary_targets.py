from __future__ import annotations

import csv
import hashlib
import json

from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

from latent_information_dynamics.datasets.target_materialization import (
    materialize_binary_targets,
)
from latent_information_dynamics.market_data.binance_csv import (
    read_klines_directory,
)
from latent_information_dynamics.targets.binary import (
    BinaryDirectionTarget,
    TieRule,
)
from latent_information_dynamics.targets.binance_reference import (
    BINANCE_BOUNDARY_SOURCE,
)


RAW_ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

PRICE_FEATURE_FILE = Path(
    "data/derived/price_baseline/"
    "btc_price_features_4h_2020_2025.csv"
)

PRICE_METADATA_FILE = Path(
    "data/derived/price_baseline/"
    "btc_price_features_4h_2020_2025.json"
)

OUTPUT_ROOT = Path(
    "data/derived/targets"
)

TARGET_FILE = (
    OUTPUT_ROOT
    / "btc_binary_targets_4h_2020_2025.csv"
)

REJECTION_FILE = (
    OUTPUT_ROOT
    / "btc_binary_targets_4h_rejections_2020_2025.csv"
)

METADATA_FILE = (
    OUTPUT_ROOT
    / "btc_binary_targets_4h_2020_2025.json"
)

HORIZON = timedelta(hours=4)
TIE_RULE = TieRule.UP

NEAR_ZERO_THRESHOLDS = {
    "1bp": 0.0001,
    "5bp": 0.0005,
    "10bp": 0.0010,
    "25bp": 0.0025,
    "50bp": 0.0050,
}


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def quantile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        raise ValueError(
            "Cannot compute quantile "
            "of empty values."
        )

    if q < 0.0 or q > 1.0:
        raise ValueError(
            "q must be between 0 and 1."
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        q
        * (len(ordered) - 1)
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = (
        position - lower
    )

    return (
        ordered[lower]
        * (1.0 - weight)
        + ordered[upper]
        * weight
    )


def class_balance(
    targets: list[
        BinaryDirectionTarget
    ],
) -> dict[str, float | int | None]:
    up = sum(
        target.outcome == 1
        for target in targets
    )

    down = sum(
        target.outcome == 0
        for target in targets
    )

    excluded = sum(
        target.outcome is None
        for target in targets
    )

    ties = sum(
        target.start_price
        == target.end_price
        for target in targets
    )

    labeled = up + down

    return {
        "count": len(targets),
        "up": up,
        "down": down,
        "excluded": excluded,
        "ties": ties,
        "up_share": (
            up / labeled
            if labeled
            else None
        ),
        "down_share": (
            down / labeled
            if labeled
            else None
        ),
    }


def main() -> None:
    print(
        "Reading accepted price decisions..."
    )

    with PRICE_FEATURE_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        decision_times = [
            datetime.fromisoformat(
                row["decision_time"]
            )
            for row in reader
        ]

    if not decision_times:
        raise RuntimeError(
            "Price feature dataset is empty."
        )

    if decision_times != sorted(
        decision_times
    ):
        raise RuntimeError(
            "Decision times are not sorted."
        )

    if len(
        set(decision_times)
    ) != len(
        decision_times
    ):
        raise RuntimeError(
            "Duplicate decision times "
            "in price feature dataset."
        )

    for decision_time in decision_times:
        if decision_time.tzinfo is None:
            raise RuntimeError(
                "Naive decision timestamp."
            )

        if (
            decision_time.minute != 0
            or decision_time.second != 0
            or decision_time.microsecond != 0
            or decision_time.hour
            not in (0, 4, 8, 12, 16, 20)
        ):
            raise RuntimeError(
                "Invalid decision boundary: "
                f"{decision_time.isoformat()}"
            )

    print(
        f"Feature-valid decisions: "
        f"{len(decision_times):,}"
    )

    print(
        "Loading frozen BTC bars..."
    )

    bars = read_klines_directory(
        RAW_ROOT
    )

    print(
        f"Loaded bars: {len(bars):,}"
    )

    print(
        "Materializing 4-hour "
        "binary targets..."
    )

    result = materialize_binary_targets(
        bars,
        decision_times=decision_times,
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

    accepted_times = [
        target.window.decision_time
        for target in result.targets
    ]

    rejected_times = [
        rejected.decision_time
        for rejected in result.rejected
    ]

    if (
        set(accepted_times)
        & set(rejected_times)
    ):
        raise RuntimeError(
            "Decision appears in both "
            "accepted and rejected targets."
        )

    if (
        set(accepted_times)
        | set(rejected_times)
        != set(decision_times)
    ):
        raise RuntimeError(
            "Not every feature decision "
            "was accounted for."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TARGET_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_time",
                "resolution_time",
                "start_price",
                "end_price",
                "arithmetic_return",
                "log_return",
                "outcome",
                "tie_rule",
                "reference_source",
                "start_available_time",
                "end_available_time",
            ],
        )

        writer.writeheader()

        for target in result.targets:
            writer.writerow(
                {
                    "decision_time":
                        target.window.decision_time.isoformat(),
                    "resolution_time":
                        target.window.resolution_time.isoformat(),
                    "start_price":
                        target.start_price,
                    "end_price":
                        target.end_price,
                    "arithmetic_return":
                        target.arithmetic_return,
                    "log_return":
                        target.log_return,
                    "outcome":
                        target.outcome,
                    "tie_rule":
                        target.tie_rule.value,
                    "reference_source":
                        target.start_reference.source,
                    "start_available_time":
                        target.start_reference.available_time.isoformat(),
                    "end_available_time":
                        target.end_reference.available_time.isoformat(),
                }
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

        for rejected in result.rejected:
            writer.writerow(
                {
                    "decision_time":
                        rejected.decision_time.isoformat(),
                    "resolution_time":
                        (
                            rejected.decision_time
                            + HORIZON
                        ).isoformat(),
                    "reason":
                        rejected.reason,
                }
            )

    targets = list(
        result.targets
    )

    returns = [
        target.arithmetic_return
        for target in targets
    ]

    overall_balance = class_balance(
        targets
    )

    near_zero = {}

    for name, threshold in (
        NEAR_ZERO_THRESHOLDS.items()
    ):
        count = sum(
            abs(value) <= threshold
            for value in returns
        )

        near_zero[name] = {
            "absolute_return_threshold":
                threshold,
            "count":
                count,
            "share":
                (
                    count
                    / len(returns)
                    if returns
                    else None
                ),
        }

    yearly_targets: dict[
        int,
        list[BinaryDirectionTarget],
    ] = {}

    hourly_targets: dict[
        int,
        list[BinaryDirectionTarget],
    ] = {}

    for target in targets:
        year = (
            target.window.decision_time.year
        )

        hour = (
            target.window.decision_time.hour
        )

        yearly_targets.setdefault(
            year,
            [],
        ).append(target)

        hourly_targets.setdefault(
            hour,
            [],
        ).append(target)

    yearly_balance = {
        str(year):
            class_balance(
                yearly_targets[year]
            )
        for year in sorted(
            yearly_targets
        )
    }

    hourly_balance = {
        f"{hour:02d}:00":
            class_balance(
                hourly_targets[hour]
            )
        for hour in sorted(
            hourly_targets
        )
    }

    return_summary = {
        "minimum":
            min(returns),
        "q01":
            quantile(
                returns,
                0.01,
            ),
        "q05":
            quantile(
                returns,
                0.05,
            ),
        "q25":
            quantile(
                returns,
                0.25,
            ),
        "median":
            median(returns),
        "mean":
            mean(returns),
        "q75":
            quantile(
                returns,
                0.75,
            ),
        "q95":
            quantile(
                returns,
                0.95,
            ),
        "q99":
            quantile(
                returns,
                0.99,
            ),
        "maximum":
            max(returns),
    }

    price_metadata = json.loads(
        PRICE_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    metadata = {
        "dataset":
            "btc-binary-targets-4h-2020-2025",
        "label_definition":
            (
                "outcome = 1 if "
                "end_price >= start_price, "
                "otherwise 0"
            ),
        "horizon_hours":
            4,
        "tie_rule":
            TIE_RULE.value,
        "reference_source":
            BINANCE_BOUNDARY_SOURCE,
        "reference_definition":
            (
                "Close of the completed "
                "5-minute Binance BTCUSDT "
                "bar immediately preceding "
                "each UTC decision boundary. "
                "Missing or irregular "
                "reference bars are rejected."
            ),
        "requested_feature_decisions":
            result.requested_decisions,
        "accepted_targets":
            result.accepted_count,
        "rejected_targets":
            result.rejected_count,
        "acceptance_rate":
            (
                result.accepted_count
                / result.requested_decisions
            ),
        "class_balance":
            overall_balance,
        "near_zero_returns":
            near_zero,
        "arithmetic_return_summary":
            return_summary,
        "yearly_class_balance":
            yearly_balance,
        "decision_hour_class_balance":
            hourly_balance,
        "source_price_feature_dataset":
            price_metadata["dataset"],
        "source_price_feature_sha256":
            sha256_file(
                PRICE_FEATURE_FILE
            ),
        "source_raw_dataset_id":
            price_metadata[
                "source_dataset_id"
            ],
        "source_raw_dataset_fingerprint_sha256":
            price_metadata[
                "source_dataset_fingerprint_sha256"
            ],
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
        "BTC 4-HOUR TARGET MATERIALIZATION"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Requested decisions: "
        f"{result.requested_decisions:,}"
    )

    print(
        f"Accepted targets:    "
        f"{result.accepted_count:,}"
    )

    print(
        f"Rejected targets:    "
        f"{result.rejected_count:,}"
    )

    print(
        f"UP:                  "
        f"{overall_balance["up"]:,}"
    )

    print(
        f"DOWN:                "
        f"{overall_balance["down"]:,}"
    )

    print(
        f"Ties:                "
        f"{overall_balance["ties"]:,}"
    )

    print(
        f"UP share:            "
        f"{overall_balance["up_share"]:.4%}"
    )

    print()

    print(
        "Near-zero absolute returns:"
    )

    for name in (
        "1bp",
        "5bp",
        "10bp",
        "25bp",
        "50bp",
    ):
        item = near_zero[name]

        print(
            f"  {name:>4}: "
            f"{item["count"]:,} "
            f"({item["share"]:.4%})"
        )

    print()
    print(
        f"Targets  -> {TARGET_FILE}"
    )
    print(
        f"Rejected -> {REJECTION_FILE}"
    )
    print(
        f"Metadata -> {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()
