from __future__ import annotations

import csv
import json
import math

from pathlib import Path
from statistics import mean


PRICE_FILE = Path(
    "data/derived/experiments/price_baseline/"
    "m0_vs_mprice_predictions.csv"
)

COMBINED_FILE = Path(
    "data/derived/experiments/combined_baseline/"
    "m0_vs_mcombined_predictions.csv"
)

OUTPUT_FILE = Path(
    "data/derived/experiments/combined_baseline/"
    "mprice_vs_mcombined_significance.json"
)

EXPECTED_ROWS = 10840
HAC_LAGS = (0, 1, 6, 12, 42)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def clip_probability(value: float) -> float:
    epsilon = 1e-15
    return min(
        max(value, epsilon),
        1.0 - epsilon,
    )


def log_loss_one(
    target: int,
    probability: float,
) -> float:
    p = clip_probability(probability)

    return -(
        target * math.log(p)
        + (1 - target)
        * math.log(1.0 - p)
    )


def brier_one(
    target: int,
    probability: float,
) -> float:
    return (
        probability - target
    ) ** 2


def hac_statistics(
    values: list[float],
    lag: int,
) -> dict[str, float | int]:
    n = len(values)
    average = mean(values)

    centered = [
        value - average
        for value in values
    ]

    long_run_variance = (
        sum(
            value * value
            for value in centered
        )
        / n
    )

    for k in range(1, lag + 1):
        covariance = (
            sum(
                centered[t]
                * centered[t - k]
                for t in range(k, n)
            )
            / n
        )

        weight = (
            1.0
            - k / (lag + 1.0)
        )

        long_run_variance += (
            2.0
            * weight
            * covariance
        )

    if long_run_variance <= 0.0:
        raise RuntimeError(
            "Non-positive HAC long-run variance."
        )

    standard_error = math.sqrt(
        long_run_variance / n
    )

    t_statistic = (
        average / standard_error
    )

    p_value = math.erfc(
        abs(t_statistic)
        / math.sqrt(2.0)
    )

    return {
        "lag": lag,
        "mean_difference":
            average,
        "hac_standard_error":
            standard_error,
        "t_statistic":
            t_statistic,
        "two_sided_asymptotic_p_value":
            p_value,
    }


def main() -> None:
    price_rows = read_rows(
        PRICE_FILE
    )

    combined_rows = read_rows(
        COMBINED_FILE
    )

    if len(price_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected price prediction count."
        )

    if len(combined_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected combined prediction count."
        )

    price_log_losses = []
    combined_log_losses = []

    price_briers = []
    combined_briers = []

    price_correct = 0
    combined_correct = 0

    first_decision = None
    last_decision = None

    for index, (
        price_row,
        combined_row,
    ) in enumerate(
        zip(
            price_rows,
            combined_rows,
            strict=True,
        )
    ):
        for key in (
            "fold",
            "decision_time",
            "target",
        ):
            if (
                price_row[key]
                != combined_row[key]
            ):
                raise RuntimeError(
                    "OOS alignment failure at row "
                    + str(index)
                    + " for "
                    + key
                )

        price_m0 = float(
            price_row[
                "m0_probability"
            ]
        )

        combined_m0 = float(
            combined_row[
                "m0_probability"
            ]
        )

        if (
            abs(
                price_m0
                - combined_m0
            )
            > 1e-15
        ):
            raise RuntimeError(
                "M0 probabilities differ at row "
                + str(index)
            )

        target = int(
            price_row[
                "target"
            ]
        )

        price_probability = float(
            price_row[
                "mprice_probability"
            ]
        )

        combined_probability = float(
            combined_row[
                "mcombined_probability"
            ]
        )

        price_log_losses.append(
            log_loss_one(
                target,
                price_probability,
            )
        )

        combined_log_losses.append(
            log_loss_one(
                target,
                combined_probability,
            )
        )

        price_briers.append(
            brier_one(
                target,
                price_probability,
            )
        )

        combined_briers.append(
            brier_one(
                target,
                combined_probability,
            )
        )

        price_correct += int(
            int(
                price_probability
                >= 0.5
            )
            == target
        )

        combined_correct += int(
            int(
                combined_probability
                >= 0.5
            )
            == target
        )

        if first_decision is None:
            first_decision = (
                price_row[
                    "decision_time"
                ]
            )

        last_decision = (
            price_row[
                "decision_time"
            ]
        )

    log_differences = [
        combined - price
        for combined, price
        in zip(
            combined_log_losses,
            price_log_losses,
            strict=True,
        )
    ]

    brier_differences = [
        combined - price
        for combined, price
        in zip(
            combined_briers,
            price_briers,
            strict=True,
        )
    ]

    n = len(price_rows)

    result = {
        "comparison":
            "mprice-vs-mcombined-4h",
        "oos_observations":
            n,
        "first_oos_decision":
            first_decision,
        "last_oos_decision":
            last_decision,
        "alignment":
            "exact",
        "difference_definition":
            "combined_minus_price",
        "negative_difference_means":
            "combined_model_better",
        "mprice": {
            "log_loss":
                mean(
                    price_log_losses
                ),
            "brier_score":
                mean(
                    price_briers
                ),
            "accuracy":
                price_correct / n,
        },
        "mcombined": {
            "log_loss":
                mean(
                    combined_log_losses
                ),
            "brier_score":
                mean(
                    combined_briers
                ),
            "accuracy":
                combined_correct / n,
        },
        "raw_differences": {
            "log_loss":
                mean(
                    log_differences
                ),
            "brier_score":
                mean(
                    brier_differences
                ),
            "accuracy":
                (
                    combined_correct
                    - price_correct
                )
                / n,
        },
        "log_loss_hac": [
            hac_statistics(
                log_differences,
                lag,
            )
            for lag in HAC_LAGS
        ],
        "brier_hac": [
            hac_statistics(
                brier_differences,
                lag,
            )
            for lag in HAC_LAGS
        ],
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "M_PRICE VS M_COMBINED"
    )
    print(
        "---------------------"
    )
    print(
        "Exact OOS alignment: PASS"
    )
    print(
        "Observations:",
        f"{n:,}",
    )
    print()

    print(
        "M_PRICE log loss:",
        f"{mean(price_log_losses):.8f}",
    )
    print(
        "M_COMBINED log loss:",
        f"{mean(combined_log_losses):.8f}",
    )
    print(
        "Combined - Price:",
        f"{mean(log_differences):+.8f}",
    )
    print()

    print(
        "M_PRICE Brier:",
        f"{mean(price_briers):.8f}",
    )
    print(
        "M_COMBINED Brier:",
        f"{mean(combined_briers):.8f}",
    )
    print(
        "Combined - Price:",
        f"{mean(brier_differences):+.8f}",
    )
    print()

    print(
        "M_PRICE accuracy:",
        f"{price_correct / n:.4%}",
    )
    print(
        "M_COMBINED accuracy:",
        f"{combined_correct / n:.4%}",
    )
    print(
        "Accuracy delta:",
        f"{(combined_correct - price_correct) / n:+.4%}",
    )

    print()
    print(
        "HAC LOG-LOSS DIFFERENCE"
    )

    for item in result[
        "log_loss_hac"
    ]:
        lag = item["lag"]
        t_value = item[
            "t_statistic"
        ]
        p_value = item[
            "two_sided_asymptotic_p_value"
        ]

        print(
            "lag",
            lag,
            "t=",
            f"{t_value:+.4f}",
            "p=",
            f"{p_value:.6f}",
        )

    print()
    print(
        "HAC BRIER DIFFERENCE"
    )

    for item in result[
        "brier_hac"
    ]:
        lag = item["lag"]
        t_value = item[
            "t_statistic"
        ]
        p_value = item[
            "two_sided_asymptotic_p_value"
        ]

        print(
            "lag",
            lag,
            "t=",
            f"{t_value:+.4f}",
            "p=",
            f"{p_value:.6f}",
        )

    print()
    print(
        "Results ->",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
