from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from latent_information_dynamics.features.kline_trade_flow import (
    KlineTradeFlowIndex,
)
from latent_information_dynamics.market_data.binance_aggtrades import (
    iter_binance_aggtrade_zip,
)
from latent_information_dynamics.market_data.binance_csv import (
    read_klines_csv,
)


UTC = timezone.utc

KLINE_FILE = Path(
    "data/raw/binance/BTCUSDT/5m/2024-01.csv"
)

AGGTRADE_FILE = Path(
    "data/raw/binance/BTCUSDT/aggTrades/"
    "BTCUSDT-aggTrades-2024-01-01.zip"
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

TOLERANCE = 1e-8


def relative_error(
    observed: float,
    reference: float,
) -> float:
    denominator = max(
        abs(reference),
        1e-12,
    )

    return abs(
        observed - reference
    ) / denominator


def main() -> None:
    bars = read_klines_csv(
        KLINE_FILE
    )

    index = KlineTradeFlowIndex(
        bars
    )

    event_results = {}

    decision_index = 0
    start_time = datetime(
        2024,
        1,
        1,
        0,
        tzinfo=UTC,
    )

    total_quantity = 0.0
    signed_quantity = 0.0
    event_count = 0

    for trade in iter_binance_aggtrade_zip(
        AGGTRADE_FILE
    ):
        if decision_index >= len(
            DECISION_TIMES
        ):
            break

        decision_time = (
            DECISION_TIMES[
                decision_index
            ]
        )

        while (
            trade.receive_time
            >= decision_time
        ):
            event_results[
                decision_time
            ] = {
                "total_quantity":
                    total_quantity,
                "signed_quantity":
                    signed_quantity,
                "event_count":
                    event_count,
            }

            start_time = decision_time
            decision_index += 1

            total_quantity = 0.0
            signed_quantity = 0.0
            event_count = 0

            if decision_index >= len(
                DECISION_TIMES
            ):
                break

            decision_time = (
                DECISION_TIMES[
                    decision_index
                ]
            )

        if decision_index >= len(
            DECISION_TIMES
        ):
            break

        if (
            start_time
            <= trade.receive_time
            < decision_time
        ):
            total_quantity += (
                trade.size
            )

            signed_quantity += (
                trade.signed_size
            )

            event_count += 1

    if decision_index < len(
        DECISION_TIMES
    ):
        raise RuntimeError(
            "aggTrades archive ended before "
            "all validation windows were completed."
        )

    print(
        "KLINE VS RAW AGGTRADE VALIDATION"
    )
    print(
        "--------------------------------"
    )

    passed = True

    for decision_time in DECISION_TIMES:
        kline = index.summarize(
            decision_time=
                decision_time,
            label="4h",
            window=__import__("datetime").timedelta(hours=4),
        )

        event = event_results[
            decision_time
        ]

        raw_total = (
            event[
                "total_quantity"
            ]
        )

        raw_signed = (
            event[
                "signed_quantity"
            ]
        )

        raw_imbalance = (
            raw_signed
            / raw_total
            if raw_total > 0
            else 0.0
        )

        total_error = relative_error(
            kline.total_quantity,
            raw_total,
        )

        signed_absolute_error = abs(
            kline.signed_quantity
            - raw_signed
        )

        imbalance_error = abs(
            kline.quantity_imbalance
            - raw_imbalance
        )

        window_passed = (
            total_error <= TOLERANCE
            and signed_absolute_error
            <= TOLERANCE
            and imbalance_error
            <= TOLERANCE
        )

        passed = (
            passed
            and window_passed
        )

        print()
        print(
            decision_time.isoformat()
        )

        print(
            f"  raw events:        "
            f"{event["event_count"]:,}"
        )

        print(
            f"  kline quantity:    "
            f"{kline.total_quantity:.10f}"
        )

        print(
            f"  raw quantity:      "
            f"{raw_total:.10f}"
        )

        print(
            f"  quantity rel err:  "
            f"{total_error:.3e}"
        )

        print(
            f"  kline signed qty:  "
            f"{kline.signed_quantity:.10f}"
        )

        print(
            f"  raw signed qty:    "
            f"{raw_signed:.10f}"
        )

        print(
            f"  signed abs err:    "
            f"{signed_absolute_error:.3e}"
        )

        print(
            f"  kline imbalance:   "
            f"{kline.quantity_imbalance:+.10f}"
        )

        print(
            f"  raw imbalance:     "
            f"{raw_imbalance:+.10f}"
        )

        print(
            f"  imbalance abs err: "
            f"{imbalance_error:.3e}"
        )

        print(
            f"  result:            "
            f"{'PASS' if window_passed else 'FAIL'}"
        )

    print()
    print(
        "OVERALL:",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
