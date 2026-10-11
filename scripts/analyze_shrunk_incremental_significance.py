from __future__ import annotations

import csv
import json
import math

from pathlib import Path
from statistics import mean
from math import erf, sqrt


INPUT_FILE = Path(
    "data/derived/experiments/"
    "probability_shrinkage/"
    "prequential_probability_shrinkage_predictions.csv"
)

OUTPUT_FILE = Path(
    "data/derived/experiments/"
    "probability_shrinkage/"
    "shrunk_price_vs_combined_significance.json"
)

HAC_LAGS = (
    0,
    1,
    6,
    12,
    42,
)

EPSILON = 1e-15


def clip_probability(
    probability: float,
) -> float:
    return min(
        max(
            probability,
            EPSILON,
        ),
        1.0 - EPSILON,
    )


def log_loss_observation(
    target: int,
    probability: float,
) -> float:
    p = clip_probability(
        probability
    )

    return -(
        target * math.log(p)
        + (1 - target)
        * math.log(1.0 - p)
    )


def brier_observation(
    target: int,
    probability: float,
) -> float:
    return (
        probability - target
    ) ** 2


def accuracy_observation(
    target: int,
    probability: float,
) -> int:
    return int(
        (
            probability >= 0.5
        )
        == target
    )


def normal_two_sided_p(
    statistic: float,
) -> float:
    z = abs(
        statistic
    )

    cdf = 0.5 * (
        1.0
        + erf(
            z / sqrt(2.0)
        )
    )

    return 2.0 * (
        1.0 - cdf
    )


def hac_standard_error(
    values: list[float],
    lag: int,
) -> float:
    n = len(
        values
    )

    if n < 2:
        raise RuntimeError(
            "Need at least two observations."
        )

    if lag < 0:
        raise ValueError(
            "HAC lag cannot be negative."
        )

    if lag >= n:
        raise ValueError(
            "HAC lag must be smaller "
            "than sample size."
        )

    mu = mean(
        values
    )

    centered = [
        value - mu
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

    for ell in range(
        1,
        lag + 1,
    ):
        covariance = (
            sum(
                centered[t]
                * centered[t - ell]
                for t in range(
                    ell,
                    n,
                )
            )
            / n
        )

        weight = (
            1.0
            - ell
            / (lag + 1.0)
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


def load_rows() -> list[dict[str, object]]:
    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        required = {
            "fold",
            "decision_time",
            "target",
            "mprice_shrunk_probability",
            "mcombined_shrunk_probability",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise RuntimeError(
                "Missing required columns: "
                f"{sorted(missing)}"
            )

        for raw in reader:
            rows.append(
                {
                    "fold":
                        int(
                            raw["fold"]
                        ),
                    "decision_time":
                        raw[
                            "decision_time"
                        ],
                    "target":
                        int(
                            raw["target"]
                        ),
                    "price_probability":
                        float(
                            raw[
                                "mprice_shrunk_probability"
                            ]
                        ),
                    "combined_probability":
                        float(
                            raw[
                                "mcombined_shrunk_probability"
                            ]
                        ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No predictions loaded."
        )

    return rows


def evaluate_sample(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    price_logloss = []
    combined_logloss = []

    price_brier = []
    combined_brier = []

    price_accuracy = []
    combined_accuracy = []

    for row in rows:
        target = int(
            row["target"]
        )

        price_probability = float(
            row[
                "price_probability"
            ]
        )

        combined_probability = float(
            row[
                "combined_probability"
            ]
        )

        price_logloss.append(
            log_loss_observation(
                target,
                price_probability,
            )
        )

        combined_logloss.append(
            log_loss_observation(
                target,
                combined_probability,
            )
        )

        price_brier.append(
            brier_observation(
                target,
                price_probability,
            )
        )

        combined_brier.append(
            brier_observation(
                target,
                combined_probability,
            )
        )

        price_accuracy.append(
            accuracy_observation(
                target,
                price_probability,
            )
        )

        combined_accuracy.append(
            accuracy_observation(
                target,
                combined_probability,
            )
        )

    logloss_difference = [
        combined - price
        for price, combined
        in zip(
            price_logloss,
            combined_logloss,
            strict=True,
        )
    ]

    brier_difference = [
        combined - price
        for price, combined
        in zip(
            price_brier,
            combined_brier,
            strict=True,
        )
    ]

    hac = {
        "log_loss": {},
        "brier": {},
    }

    for lag in HAC_LAGS:
        for name, difference in (
            (
                "log_loss",
                logloss_difference,
            ),
            (
                "brier",
                brier_difference,
            ),
        ):
            se = hac_standard_error(
                difference,
                lag,
            )

            difference_mean = mean(
                difference
            )

            if se == 0.0:
                statistic = (
                    0.0
                    if difference_mean == 0.0
                    else math.copysign(
                        math.inf,
                        difference_mean,
                    )
                )
            else:
                statistic = (
                    difference_mean
                    / se
                )

            p_value = (
                0.0
                if math.isinf(
                    statistic
                )
                else normal_two_sided_p(
                    statistic
                )
            )

            hac[
                name
            ][
                str(lag)
            ] = {
                "standard_error":
                    se,
                "t_statistic":
                    statistic,
                "p_value":
                    p_value,
            }

    folds = sorted(
        {
            int(
                row["fold"]
            )
            for row in rows
        }
    )

    fold_results = []

    for fold in folds:
        fold_rows = [
            row
            for row in rows
            if int(
                row["fold"]
            ) == fold
        ]

        fold_price_ll = mean(
            log_loss_observation(
                int(
                    row["target"]
                ),
                float(
                    row[
                        "price_probability"
                    ]
                ),
            )
            for row in fold_rows
        )

        fold_combined_ll = mean(
            log_loss_observation(
                int(
                    row["target"]
                ),
                float(
                    row[
                        "combined_probability"
                    ]
                ),
            )
            for row in fold_rows
        )

        fold_price_bs = mean(
            brier_observation(
                int(
                    row["target"]
                ),
                float(
                    row[
                        "price_probability"
                    ]
                ),
            )
            for row in fold_rows
        )

        fold_combined_bs = mean(
            brier_observation(
                int(
                    row["target"]
                ),
                float(
                    row[
                        "combined_probability"
                    ]
                ),
            )
            for row in fold_rows
        )

        fold_results.append(
            {
                "fold":
                    fold,
                "observations":
                    len(
                        fold_rows
                    ),
                "log_loss_delta_combined_minus_price":
                    (
                        fold_combined_ll
                        - fold_price_ll
                    ),
                "brier_delta_combined_minus_price":
                    (
                        fold_combined_bs
                        - fold_price_bs
                    ),
            }
        )

    return {
        "observations":
            len(
                rows
            ),
        "folds":
            len(
                folds
            ),
        "price": {
            "log_loss":
                mean(
                    price_logloss
                ),
            "brier_score":
                mean(
                    price_brier
                ),
            "accuracy":
                mean(
                    price_accuracy
                ),
        },
        "combined": {
            "log_loss":
                mean(
                    combined_logloss
                ),
            "brier_score":
                mean(
                    combined_brier
                ),
            "accuracy":
                mean(
                    combined_accuracy
                ),
        },
        "combined_minus_price": {
            "log_loss":
                mean(
                    logloss_difference
                ),
            "brier_score":
                mean(
                    brier_difference
                ),
            "accuracy":
                (
                    mean(
                        combined_accuracy
                    )
                    - mean(
                        price_accuracy
                    )
                ),
        },
        "combined_fold_wins": {
            "log_loss":
                sum(
                    row[
                        "log_loss_delta_combined_minus_price"
                    ] < 0.0
                    for row in fold_results
                ),
            "brier":
                sum(
                    row[
                        "brier_delta_combined_minus_price"
                    ] < 0.0
                    for row in fold_results
                ),
        },
        "hac":
            hac,
        "fold_results":
            fold_results,
    }


def print_result(
    title: str,
    result: dict[str, object],
) -> None:
    print(
        title
    )
    print(
        "-" * len(title)
    )

    print(
        f"Observations: "
        f"{result['observations']:,}"
    )

    print(
        f"Folds: "
        f"{result['folds']}"
    )

    print()

    price = result[
        "price"
    ]

    combined = result[
        "combined"
    ]

    difference = result[
        "combined_minus_price"
    ]

    print(
        "M_PRICE_SHRUNK log loss: "
        f"{price['log_loss']:.8f}"
    )

    print(
        "M_COMBINED_SHRUNK log loss: "
        f"{combined['log_loss']:.8f}"
    )

    print(
        "Combined - Price: "
        f"{difference['log_loss']:+.8f}"
    )

    print()

    print(
        "M_PRICE_SHRUNK Brier: "
        f"{price['brier_score']:.8f}"
    )

    print(
        "M_COMBINED_SHRUNK Brier: "
        f"{combined['brier_score']:.8f}"
    )

    print(
        "Combined - Price: "
        f"{difference['brier_score']:+.8f}"
    )

    print()

    print(
        "M_PRICE_SHRUNK accuracy: "
        f"{price['accuracy']:.4%}"
    )

    print(
        "M_COMBINED_SHRUNK accuracy: "
        f"{combined['accuracy']:.4%}"
    )

    print(
        "Accuracy delta: "
        f"{difference['accuracy']:+.4%}"
    )

    print()

    wins = result[
        "combined_fold_wins"
    ]

    print(
        "Combined fold wins: "
        f"LL {wins['log_loss']}/"
        f"{result['folds']}, "
        f"Brier {wins['brier']}/"
        f"{result['folds']}"
    )

    print()

    for metric in (
        "log_loss",
        "brier",
    ):
        print(
            f"HAC {metric.upper()}"
        )

        for lag in HAC_LAGS:
            item = result[
                "hac"
            ][
                metric
            ][
                str(lag)
            ]

            print(
                f"lag {lag:2d}: "
                f"t="
                f"{item['t_statistic']:+.4f} "
                f"p="
                f"{item['p_value']:.6f}"
            )

        print()


def main() -> None:
    rows = load_rows()

    all_oos = evaluate_sample(
        rows
    )

    post_cold_start = evaluate_sample(
        [
            row
            for row in rows
            if int(
                row["fold"]
            ) > 0
        ]
    )

    results = {
        "experiment":
            "shrunk-price-vs-combined-significance",
        "difference_definition":
            "M_COMBINED_SHRUNK minus M_PRICE_SHRUNK",
        "all_oos":
            all_oos,
        "post_cold_start":
            post_cold_start,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )

    print_result(
        "ALL OOS — SHRUNK PRICE VS COMBINED",
        all_oos,
    )

    print_result(
        "POST-COLD-START — SHRUNK PRICE VS COMBINED",
        post_cold_start,
    )

    print(
        "Results ->",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
