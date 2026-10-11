from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PRICE_FILE = Path(
    "data/derived/supervised/"
    "btc_price_supervised_4h_2020_2025.csv"
)

FLOW_FILE = Path(
    "data/derived/supervised/"
    "btc_trade_flow_supervised_4h_2020_2025.csv"
)

PRICE_METADATA = Path(
    "data/derived/supervised/"
    "btc_price_supervised_4h_2020_2025.json"
)

FLOW_METADATA = Path(
    "data/derived/supervised/"
    "btc_trade_flow_supervised_4h_2020_2025.json"
)

OUTPUT_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_4h_2020_2025.csv"
)

OUTPUT_METADATA = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_4h_2020_2025.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    price_meta = json.loads(
        PRICE_METADATA.read_text(
            encoding="utf-8"
        )
    )

    flow_meta = json.loads(
        FLOW_METADATA.read_text(
            encoding="utf-8"
        )
    )

    price_features = tuple(
        price_meta["features"]
    )

    flow_features = tuple(
        flow_meta["features"]
    )

    if len(price_features) != 11:
        raise RuntimeError(
            "Expected 11 frozen price features."
        )

    if len(flow_features) != 20:
        raise RuntimeError(
            "Expected 20 frozen trade-flow features."
        )

    overlap = (
        set(price_features)
        & set(flow_features)
    )

    if overlap:
        raise RuntimeError(
            "Feature-name overlap: "
            + repr(sorted(overlap))
        )

    price_rows = read_csv(
        PRICE_FILE
    )

    flow_rows = read_csv(
        FLOW_FILE
    )

    if len(price_rows) != len(flow_rows):
        raise RuntimeError(
            "Dataset row counts differ."
        )

    combined_rows = []

    keys = (
        "decision_time",
        "resolution_time",
        "label_available_time",
        "target",
        "realized_return",
        "realized_log_return",
    )

    for index, (
        price_row,
        flow_row,
    ) in enumerate(
        zip(
            price_rows,
            flow_rows,
            strict=True,
        )
    ):
        for key in keys:
            if (
                price_row[key]
                != flow_row[key]
            ):
                raise RuntimeError(
                    f"Row {index}: "
                    f"{key} differs."
                )

        combined = {
            key: price_row[key]
            for key in keys
        }

        for name in price_features:
            combined[name] = (
                price_row[name]
            )

        for name in flow_features:
            combined[name] = (
                flow_row[name]
            )

        combined_rows.append(
            combined
        )

    if len(combined_rows) != 13030:
        raise RuntimeError(
            "Expected 13,030 aligned rows."
        )

    feature_names = (
        *price_features,
        *flow_features,
    )

    fields = [
        *keys,
        *feature_names,
    ]

    OUTPUT_FILE.parent.mkdir(
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
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            combined_rows
        )

    metadata = {
        "dataset":
            "btc-combined-supervised-4h-2020-2025",
        "rows":
            len(combined_rows),
        "feature_count":
            len(feature_names),
        "price_feature_count":
            len(price_features),
        "trade_flow_feature_count":
            len(flow_features),
        "features":
            list(feature_names),
        "price_features":
            list(price_features),
        "trade_flow_features":
            list(flow_features),
        "price_dataset_sha256":
            sha256_file(PRICE_FILE),
        "trade_flow_dataset_sha256":
            sha256_file(FLOW_FILE),
        "strict_timestamp_alignment":
            True,
        "supervised_csv_sha256":
            None,
    }

    metadata[
        "supervised_csv_sha256"
    ] = sha256_file(
        OUTPUT_FILE
    )

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "COMBINED SUPERVISED DATASET"
    )
    print(
        "---------------------------"
    )
    print(
        "Rows:",
        f"{len(combined_rows):,}",
    )
    print(
        "Price features:",
        len(price_features),
    )
    print(
        "Trade-flow features:",
        len(flow_features),
    )
    print(
        "Total features:",
        len(feature_names),
    )
    print(
        "Timestamp/target alignment: PASS"
    )
    print(
        "SHA256:",
        sha256_file(
            OUTPUT_FILE
        ),
    )
    print(
        "Output:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
