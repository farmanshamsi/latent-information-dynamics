from __future__ import annotations

import csv

from datetime import datetime, timezone
from pathlib import Path

from latent_information_dynamics.features.trade_flow import (
    DEFAULT_TRADE_FLOW_WINDOWS,
    MultiHorizonTradeFlowEngine,
)
from latent_information_dynamics.market_data.binance_aggtrades import (
    iter_binance_aggtrade_zip,
)


UTC = timezone.utc

INPUT = Path(
    "data/raw/binance/BTCUSDT/aggTrades/"
    "BTCUSDT-aggTrades-2024-01-01.zip"
)

OUTPUT = Path(
    "data/derived/trade_flow/"
    "BTCUSDT-trade-flow-2024-01-01.csv"
)

DECISION_TIMES = tuple(
    datetime(
        2024,
        1,
        1,
        hour,
        tzinfo=UTC,
    )
    for hour in (
        4,
        8,
        12,
        16,
        20,
    )
)


def main() -> None:
    engine = (
        MultiHorizonTradeFlowEngine()
    )

    trades = (
        iter_binance_aggtrade_zip(
            INPUT
        )
    )

    current = next(
        trades,
        None,
    )

    rows = []

    for decision_time in DECISION_TIMES:
        processed = 0

        # Conservative boundary:
        # information timestamped exactly at
        # decision_time is NOT used.
        while (
            current is not None
            and current.receive_time
            < decision_time
        ):
            engine.on_trade(
                current
            )

            processed += 1

            current = next(
                trades,
                None,
            )

        summaries = (
            engine.summaries(
                as_of=decision_time
            )
        )

        row = {
            "decision_time":
                decision_time.isoformat(),
        }

        for label in (
            DEFAULT_TRADE_FLOW_WINDOWS
        ):
            summary = summaries[
                label
            ]

            row[
                f"aggregate_trade_count_{label}"
            ] = (
                summary.aggregate_trade_count
            )

            row[
                f"total_quantity_{label}"
            ] = (
                summary.total_quantity
            )

            row[
                f"signed_quantity_{label}"
            ] = (
                summary.signed_quantity
            )

            row[
                f"quantity_imbalance_{label}"
            ] = (
                summary.quantity_imbalance
            )

            row[
                f"count_imbalance_{label}"
            ] = (
                summary.count_imbalance
            )

        rows.append(
            row
        )

        print(
            decision_time.isoformat(),
            "processed_since_previous=",
            f"{processed:,}",
            "4h_trades=",
            f"{summaries["4h"].aggregate_trade_count:,}",
            "4h_quantity_imbalance=",
            f"{summaries["4h"].quantity_imbalance:+.6f}",
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"Rows: {len(rows)}"
    )

    print(
        f"Features per row: "
        f"{len(rows[0]) - 1}"
    )

    print(
        f"Saved -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()
