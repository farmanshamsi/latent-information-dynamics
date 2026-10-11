from __future__ import annotations

import csv
import math

from pathlib import Path


PREDICTIONS_FILE = Path(
    "data/derived/experiments/price_baseline/"
    "m0_vs_mprice_predictions.csv"
)

HAC_LAGS = (
    0,
    1,
    6,
    12,
    42,
)


def log_loss_one(
    target: int,
    probability: float,
) -> float:
    epsilon = 1e-12

    probability = min(
        1.0 - epsilon,
        max(
            epsilon,
            probability,
        ),
    )

    return -(
        target
        * math.log(
            probability
        )
        + (
            1 - target
        )
        * math.log(
            1.0 - probability
        )
    )


def brier_one(
    target: int,
    probability: float,
) -> float:
    return (
        probability
        - target
    ) ** 2


def mean(
    values: list[float],
) -> float:
    return (
        sum(values)
        / len(values)
    )


def newey_west_se(
    values: list[float],
    lag: int,
) -> float:
    n = len(values)
    average = mean(values)

    centered = [
        value - average
        for value in values
    ]

    gamma0 = (
        sum(
            value * value
            for value in centered
        )
        / n
    )

    long_run_variance = gamma0

    for k in range(
        1,
        lag + 1,
    ):
        covariance = (
            sum(
                centered[t]
                * centered[t - k]
                for t in range(
                    k,
                    n,
                )
            )
            / n
        )

        weight = (
            1.0
            - k
            / (
                lag + 1
            )
        )

        long_run_variance += (
            2.0
            * weight
            * covariance
        )

    long_run_variance = max(
        long_run_variance,
        0.0,
    )

    return math.sqrt(
        long_run_variance
        / n
    )


def main() -> None:
    log_differences = []
    brier_differences = []

    with PREDICTIONS_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            target = int(
                row["target"]
            )

            m0 = float(
                row[
                    "m0_probability"
                ]
            )

            mprice = float(
                row[
                    "mprice_probability"
                ]
            )

            log_differences.append(
                log_loss_one(
                    target,
                    mprice,
                )
                - log_loss_one(
                    target,
                    m0,
                )
            )

            brier_differences.append(
                brier_one(
                    target,
                    mprice,
                )
                - brier_one(
                    target,
                    m0,
                )
            )

    if not log_differences:
        raise RuntimeError(
            "No predictions found."
        )

    print(
        "PAIRED OOS LOSS DIAGNOSTICS"
    )
    print(
        "---------------------------"
    )

    print(
        f"Observations: "
        f"{len(log_differences):,}"
    )

    print()
    print(
        "Negative difference favors M_PRICE."
    )

    print()
    print(
        f"Mean log-loss difference: "
        f"{mean(log_differences):+.10f}"
    )

    print(
        f"Mean Brier difference:   "
        f"{mean(brier_differences):+.10f}"
    )

    print()

    for lag in HAC_LAGS:
        log_se = newey_west_se(
            log_differences,
            lag,
        )

        brier_se = newey_west_se(
            brier_differences,
            lag,
        )

        log_t = (
            mean(
                log_differences
            )
            / log_se
            if log_se > 0
            else float("nan")
        )

        brier_t = (
            mean(
                brier_differences
            )
            / brier_se
            if brier_se > 0
            else float("nan")
        )

        print(
            f"HAC lag {lag:>2}: "
            f"log-loss SE={log_se:.10f}, "
            f"t={log_t:+.4f}"
        )

        print(
            f"            "
            f"Brier   SE={brier_se:.10f}, "
            f"t={brier_t:+.4f}"
        )


if __name__ == "__main__":
    main()
