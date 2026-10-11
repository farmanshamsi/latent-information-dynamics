from __future__ import annotations

import csv
import hashlib
import json
import math

from datetime import datetime
from pathlib import Path

from latent_information_dynamics.features.kline_trade_flow import (
    DEFAULT_KLINE_FLOW_WINDOWS,
    KlineTradeFlowError,
    KlineTradeFlowIndex,
    feature_names,
)
from latent_information_dynamics.market_data.binance_csv import (
    read_klines_directory,
)


RAW_ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

TARGET_FILE = Path(
    "data/derived/targets/"
    "btc_binary_targets_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/trade_flow"
)

FEATURE_FILE = (
    OUTPUT_ROOT
    / "btc_trade_flow_features_4h_2020_2025.csv"
)

REJECTION_FILE = (
    OUTPUT_ROOT
    / "btc_trade_flow_features_4h_rejections_2020_2025.csv"
)

FEATURE_METADATA_FILE = (
    OUTPUT_ROOT
    / "btc_trade_flow_features_4h_2020_2025.json"
)

SUPERVISED_ROOT = Path(
    "data/derived/supervised"
)

SUPERVISED_FILE = (
    SUPERVISED_ROOT
    / "btc_trade_flow_supervised_4h_2020_2025.csv"
)

SUPERVISED_METADATA_FILE = (
    SUPERVISED_ROOT
    / "btc_trade_flow_supervised_4h_2020_2025.json"
)

MANIFEST_FILE = Path(
    "data/manifests/"
    "btc_trade_flow_supervised_4h_2020_2025.sha256"
)

EXPECTED_TARGET_SHA256 = (
    "c8b7866b0b5eddf774b77cf7cf923561"
    "303f18bf6eab82c519272c166d26e94d"
)

EXPECTED_TARGET_ROWS = 13030
EXPECTED_RAW_FILES = 72
EXPECTED_RAW_BARS = 630830


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
            digest.update(
                chunk
            )

    return digest.hexdigest()


def raw_source_fingerprint(
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


def load_targets() -> list[dict[str, str]]:
    observed_hash = sha256_file(
        TARGET_FILE
    )

    if (
        observed_hash
        != EXPECTED_TARGET_SHA256
    ):
        raise RuntimeError(
            "Frozen target hash differs "
            "from the validated Day 35 target."
        )

    with TARGET_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    if len(rows) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(
            "Expected exactly 13,030 "
            "frozen targets."
        )

    previous = None

    for row in rows:
        decision_time = (
            datetime.fromisoformat(
                row[
                    "decision_time"
                ]
            )
        )

        if (
            previous is not None
            and decision_time
            <= previous
        ):
            raise RuntimeError(
                "Target decision times "
                "are not strictly increasing."
            )

        previous = decision_time

    return rows


def build_feature_row(
    *,
    decision_time: datetime,
    summaries,
) -> dict[str, object]:
    row: dict[str, object] = {
        "decision_time":
            decision_time.isoformat(),
        "available_time":
            decision_time.isoformat(),
    }

    for label in (
        DEFAULT_KLINE_FLOW_WINDOWS
    ):
        summary = summaries[
            label
        ]

        if (
            summary.available_time
            != decision_time
        ):
            raise RuntimeError(
                "Feature availability differs "
                "from decision time."
            )

        row[
            "total_quantity_"
            + label
        ] = (
            summary.total_quantity
        )

        row[
            "signed_quantity_"
            + label
        ] = (
            summary.signed_quantity
        )

        row[
            "quantity_imbalance_"
            + label
        ] = (
            summary.quantity_imbalance
        )

        row[
            "trade_count_"
            + label
        ] = (
            summary.trade_count
        )

    return row


def main() -> None:
    print(
        "Loading frozen 2020-2025 "
        "Binance 5m history..."
    )

    raw_files = sorted(
        RAW_ROOT.glob(
            "????-??.csv"
        )
    )

    if len(raw_files) != EXPECTED_RAW_FILES:
        raise RuntimeError(
            "Expected exactly 72 "
            "monthly raw files."
        )

    source_fingerprint = (
        raw_source_fingerprint(
            raw_files
        )
    )

    bars = read_klines_directory(
        RAW_ROOT
    )

    if len(bars) != EXPECTED_RAW_BARS:
        raise RuntimeError(
            "Expected exactly 630,830 "
            "observed 5m bars."
        )

    print(
        "Raw files:",
        len(raw_files),
    )

    print(
        "Raw bars:",
        f"{len(bars):,}",
    )

    print(
        "Building point-in-time "
        "trade-flow index..."
    )

    index = KlineTradeFlowIndex(
        bars
    )

    targets = load_targets()

    names = feature_names()

    if len(names) != 20:
        raise RuntimeError(
            "Expected exactly 20 "
            "trade-flow features."
        )

    forbidden_tokens = (
        "price",
        "return",
        "open",
        "high",
        "low",
        "close",
        "quote",
        "notional",
    )

    for name in names:
        if any(
            token in name
            for token
            in forbidden_tokens
        ):
            raise RuntimeError(
                "Price-derived feature "
                "entered strict information set: "
                + name
            )

    feature_rows = []
    supervised_rows = []
    rejections = []

    for target_row in targets:
        decision_time = (
            datetime.fromisoformat(
                target_row[
                    "decision_time"
                ]
            )
        )

        try:
            summaries = (
                index.summaries(
                    decision_time=
                        decision_time
                )
            )

        except KlineTradeFlowError as exc:
            rejections.append(
                {
                    "decision_time":
                        decision_time.isoformat(),
                    "reason":
                        str(exc),
                }
            )

            continue

        feature_row = build_feature_row(
            decision_time=
                decision_time,
            summaries=summaries,
        )

        numeric_values = [
            float(
                feature_row[
                    name
                ]
            )
            for name in names
        ]

        if any(
            not math.isfinite(
                value
            )
            for value
            in numeric_values
        ):
            raise RuntimeError(
                "Non-finite trade-flow feature."
            )

        resolution_time = (
            datetime.fromisoformat(
                target_row[
                    "resolution_time"
                ]
            )
        )

        label_available_time = (
            datetime.fromisoformat(
                target_row[
                    "end_available_time"
                ]
            )
        )

        if (
            label_available_time
            < resolution_time
        ):
            raise RuntimeError(
                "Label availability precedes "
                "target resolution."
            )

        if (
            decision_time
            >= label_available_time
        ):
            raise RuntimeError(
                "Label is not genuinely future."
            )

        feature_rows.append(
            feature_row
        )

        supervised = {
            "decision_time":
                decision_time.isoformat(),
            "resolution_time":
                resolution_time.isoformat(),
            "label_available_time":
                label_available_time.isoformat(),
            "target":
                int(
                    target_row[
                        "outcome"
                    ]
                ),
            "realized_return":
                float(
                    target_row[
                        "arithmetic_return"
                    ]
                ),
            "realized_log_return":
                float(
                    target_row[
                        "log_return"
                    ]
                ),
        }

        for name in names:
            supervised[
                name
            ] = feature_row[
                name
            ]

        supervised_rows.append(
            supervised
        )

    if not feature_rows:
        raise RuntimeError(
            "No trade-flow features "
            "were materialized."
        )

    if (
        len(feature_rows)
        != len(
            supervised_rows
        )
    ):
        raise RuntimeError(
            "Feature and supervised "
            "row counts differ."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUPERVISED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_fields = [
        "decision_time",
        "available_time",
        *names,
    ]

    with FEATURE_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                feature_fields,
        )

        writer.writeheader()

        writer.writerows(
            feature_rows
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

        writer.writerows(
            rejections
        )

    supervised_fields = [
        "decision_time",
        "resolution_time",
        "label_available_time",
        "target",
        "realized_return",
        "realized_log_return",
        *names,
    ]

    with SUPERVISED_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                supervised_fields,
        )

        writer.writeheader()

        writer.writerows(
            supervised_rows
        )

    feature_hash = sha256_file(
        FEATURE_FILE
    )

    rejection_hash = sha256_file(
        REJECTION_FILE
    )

    supervised_hash = sha256_file(
        SUPERVISED_FILE
    )

    feature_metadata = {
        "dataset":
            "btc-trade-flow-features-4h-2020-2025",
        "source":
            "binance-spot-btcusdt-5m",
        "source_files":
            len(raw_files),
        "source_bars":
            len(bars),
        "source_fingerprint_sha256":
            source_fingerprint,
        "target_source":
            str(TARGET_FILE),
        "target_sha256":
            EXPECTED_TARGET_SHA256,
        "requested_decisions":
            len(targets),
        "accepted":
            len(feature_rows),
        "rejected":
            len(rejections),
        "feature_count":
            len(names),
        "features":
            list(names),
        "horizons":
            list(
                DEFAULT_KLINE_FLOW_WINDOWS
            ),
        "price_independent":
            True,
        "uses_ohlc_as_features":
            False,
        "uses_quote_volume_as_feature":
            False,
        "first_decision_time":
            feature_rows[
                0
            ][
                "decision_time"
            ],
        "last_decision_time":
            feature_rows[
                -1
            ][
                "decision_time"
            ],
        "feature_csv_sha256":
            feature_hash,
        "rejection_csv_sha256":
            rejection_hash,
    }

    FEATURE_METADATA_FILE.write_text(
        json.dumps(
            feature_metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    outcomes = [
        int(
            row[
                "target"
            ]
        )
        for row
        in supervised_rows
    ]

    supervised_metadata = {
        "dataset":
            "btc-trade-flow-supervised-4h-2020-2025",
        "feature_dataset":
            str(FEATURE_FILE),
        "target_dataset":
            str(TARGET_FILE),
        "rows":
            len(
                supervised_rows
            ),
        "feature_count":
            len(names),
        "features":
            list(names),
        "up":
            sum(outcomes),
        "down":
            len(outcomes)
            - sum(outcomes),
        "up_share":
            (
                sum(outcomes)
                / len(outcomes)
            ),
        "first_decision_time":
            supervised_rows[
                0
            ][
                "decision_time"
            ],
        "last_decision_time":
            supervised_rows[
                -1
            ][
                "decision_time"
            ],
        "strict_point_in_time":
            True,
        "price_independent_features":
            True,
        "supervised_csv_sha256":
            supervised_hash,
    }

    SUPERVISED_METADATA_FILE.write_text(
        json.dumps(
            supervised_metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_paths = [
        FEATURE_FILE,
        REJECTION_FILE,
        FEATURE_METADATA_FILE,
        SUPERVISED_FILE,
        SUPERVISED_METADATA_FILE,
    ]

    manifest_lines = []

    for path in manifest_paths:
        manifest_lines.append(
            sha256_file(
                path
            )
            + "  "
            + str(path)
        )

    MANIFEST_FILE.write_text(
        "\n".join(
            manifest_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "TRADE-FLOW MATERIALIZATION"
    )

    print(
        "--------------------------"
    )

    print(
        "Requested targets:",
        f"{len(targets):,}",
    )

    print(
        "Accepted features:",
        f"{len(feature_rows):,}",
    )

    print(
        "Rejected:",
        f"{len(rejections):,}",
    )

    print(
        "Feature count:",
        len(names),
    )

    print(
        "Supervised rows:",
        f"{len(supervised_rows):,}",
    )

    print(
        "UP:",
        f"{sum(outcomes):,}",
    )

    print(
        "DOWN:",
        f"{len(outcomes) - sum(outcomes):,}",
    )

    print(
        "UP share:",
        f"{sum(outcomes) / len(outcomes):.4%}",
    )

    print(
        "First:",
        supervised_rows[
            0
        ][
            "decision_time"
        ],
    )

    print(
        "Last:",
        supervised_rows[
            -1
        ][
            "decision_time"
        ],
    )

    print()
    print(
        "PRICE-INDEPENDENCE AUDIT PASSED"
    )

    print(
        "PIT AVAILABILITY AUDIT PASSED"
    )

    print()
    print(
        "Feature CSV ->",
        FEATURE_FILE,
    )

    print(
        "Supervised ->",
        SUPERVISED_FILE,
    )

    print(
        "Manifest ->",
        MANIFEST_FILE,
    )


if __name__ == "__main__":
    main()
