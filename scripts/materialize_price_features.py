from __future__ import annotations

import csv
import json

from datetime import datetime, timezone
from pathlib import Path

from latent_information_dynamics.datasets.price_materialization import (
    materialize_price_features,
)
from latent_information_dynamics.market_data.binance_csv import (
    read_klines_directory,
)


UTC = timezone.utc

RAW_ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

OUTPUT_ROOT = Path(
    "data/derived/price_baseline"
)

SOURCE_MANIFEST = Path(
    "data/manifests/binance_btcusdt_5m_2020_2025.json"
)

START = datetime(
    2020, 1, 1, 0, 0,
    tzinfo=UTC,
)

END = datetime(
    2026, 1, 1, 0, 0,
    tzinfo=UTC,
)

FEATURE_FILE = (
    OUTPUT_ROOT
    / "btc_price_features_4h_2020_2025.csv"
)

REJECTION_FILE = (
    OUTPUT_ROOT
    / "btc_price_features_4h_rejections_2020_2025.csv"
)

METADATA_FILE = (
    OUTPUT_ROOT
    / "btc_price_features_4h_2020_2025.json"
)


def main() -> None:
    print(
        "Loading frozen BTC dataset..."
    )

    bars = read_klines_directory(
        RAW_ROOT
    )

    print(
        f"Loaded bars: {len(bars):,}"
    )

    print(
        "Materializing 4-hour "
        "point-in-time price features..."
    )

    result = materialize_price_features(
        bars,
        start=START,
        end=END,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_names: list[str] = []

    if result.snapshots:
        feature_names = list(
            result.snapshots[0].features
        )

    with FEATURE_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_time",
                *feature_names,
            ],
        )

        writer.writeheader()

        for snapshot in result.snapshots:
            writer.writerow(
                {
                    "decision_time":
                        snapshot.decision_time.isoformat(),
                    **snapshot.features,
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
                "reason",
            ],
        )

        writer.writeheader()

        for rejected in result.rejected:
            writer.writerow(
                {
                    "decision_time":
                        rejected.decision_time.isoformat(),
                    "reason":
                        rejected.reason,
                }
            )

    source_manifest = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    rejection_counts: dict[str, int] = {}

    for rejected in result.rejected:
        category = (
            "missing_bar"
            if rejected.reason.startswith(
                "Missing bar"
            )
            else
            "irregular_bar"
            if rejected.reason.startswith(
                "Irregular bar"
            )
            else
            "other"
        )

        rejection_counts[category] = (
            rejection_counts.get(
                category,
                0,
            )
            + 1
        )

    metadata = {
        "dataset":
            "btc-price-baseline-4h-2020-2025",
        "source_dataset_id":
            source_manifest["dataset_id"],
        "source_dataset_fingerprint_sha256":
            source_manifest[
                "dataset_fingerprint_sha256"
            ],
        "start_time":
            START.isoformat(),
        "end_time_exclusive":
            END.isoformat(),
        "theoretical_decisions":
            result.theoretical_decisions,
        "accepted_decisions":
            result.accepted_count,
        "rejected_decisions":
            result.rejected_count,
        "rejection_counts":
            rejection_counts,
        "feature_count":
            len(feature_names),
        "features":
            feature_names,
        "policy": (
            "Features use completed 5-minute "
            "bars only. Decisions requiring "
            "missing or irregular source bars "
            "are rejected; no interpolation "
            "or forward filling is applied."
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
        "PRICE FEATURE MATERIALIZATION"
    )
    print(
        "-----------------------------"
    )
    print(
        f"Theoretical decisions: "
        f"{result.theoretical_decisions:,}"
    )
    print(
        f"Accepted decisions:    "
        f"{result.accepted_count:,}"
    )
    print(
        f"Rejected decisions:    "
        f"{result.rejected_count:,}"
    )

    for key in sorted(
        rejection_counts
    ):
        print(
            f"  {key}: "
            f"{rejection_counts[key]:,}"
        )

    print(
        f"Features per decision: "
        f"{len(feature_names)}"
    )

    print()
    print(
        f"Features → {FEATURE_FILE}"
    )
    print(
        f"Rejected → {REJECTION_FILE}"
    )
    print(
        f"Metadata → {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()
