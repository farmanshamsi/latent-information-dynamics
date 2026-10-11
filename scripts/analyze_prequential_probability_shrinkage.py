from __future__ import annotations

import csv
import json
import math

from collections import defaultdict
from pathlib import Path
from statistics import mean


INPUT_FILE = Path(
    "data/derived/experiments/"
    "baseline_information_decomposition/"
    "baseline_model_predictions_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "probability_shrinkage"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "prequential_probability_shrinkage_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "prequential_probability_shrinkage_predictions.csv"
)

ALPHA_FILE = (
    OUTPUT_ROOT
    / "prequential_probability_shrinkage_alphas.csv"
)

ALPHA_GRID = tuple(
    step / 100.0
    for step in range(101)
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


def shrink_probability(
    baseline_probability: float,
    model_probability: float,
    alpha: float,
) -> float:
    probability = (
        baseline_probability
        + alpha
        * (
            model_probability
            - baseline_probability
        )
    )

    return clip_probability(
        probability
    )


def binary_log_loss(
    target: int,
    probability: float,
) -> float:
    p = clip_probability(
        probability
    )

    return -(
        target
        * math.log(p)
        + (
            1 - target
        )
        * math.log(
            1.0 - p
        )
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
            "m0_probability",
            "mprice_probability",
            "mcombined_probability",
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
                "Input file missing columns: "
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
                    "m0_probability":
                        float(
                            raw[
                                "m0_probability"
                            ]
                        ),
                    "mprice_probability":
                        float(
                            raw[
                                "mprice_probability"
                            ]
                        ),
                    "mcombined_probability":
                        float(
                            raw[
                                "mcombined_probability"
                            ]
                        ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No baseline predictions loaded."
        )

    return rows


def group_by_fold(
    rows: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            int(
                row["fold"]
            )
        ].append(
            row
        )

    return dict(
        grouped
    )


def historical_log_loss(
    rows: list[dict[str, object]],
    model_column: str,
    alpha: float,
) -> float:
    losses = []

    for row in rows:
        probability = shrink_probability(
            float(
                row[
                    "m0_probability"
                ]
            ),
            float(
                row[
                    model_column
                ]
            ),
            alpha,
        )

        losses.append(
            binary_log_loss(
                int(
                    row["target"]
                ),
                probability,
            )
        )

    return mean(
        losses
    )


def choose_alpha(
    history: list[dict[str, object]],
    model_column: str,
) -> tuple[float, float | None]:
    if not history:
        return 1.0, None

    candidates = []

    for alpha in ALPHA_GRID:
        score = historical_log_loss(
            history,
            model_column,
            alpha,
        )

        candidates.append(
            (
                score,
                -alpha,
                alpha,
            )
        )

    score, _, alpha = min(
        candidates
    )

    return alpha, score


def build_prequential_predictions(
    rows: list[dict[str, object]],
    model_column: str,
) -> tuple[
    dict[tuple[int, str], tuple[float, float]],
    list[dict[str, object]],
]:
    grouped = group_by_fold(
        rows
    )

    history = []

    predictions = {}

    alpha_schedule = []

    folds = sorted(
        grouped
    )

    for fold in folds:
        current = grouped[
            fold
        ]

        alpha, historical_score = (
            choose_alpha(
                history,
                model_column,
            )
        )

        alpha_schedule.append(
            {
                "fold":
                    fold,
                "alpha":
                    alpha,
                "selection_observations":
                    len(
                        history
                    ),
                "selection_log_loss":
                    historical_score,
            }
        )

        for row in current:
            probability = (
                shrink_probability(
                    float(
                        row[
                            "m0_probability"
                        ]
                    ),
                    float(
                        row[
                            model_column
                        ]
                    ),
                    alpha,
                )
            )

            key = (
                int(
                    row["fold"]
                ),
                str(
                    row[
                        "decision_time"
                    ]
                ),
            )

            predictions[
                key
            ] = (
                alpha,
                probability,
            )

        # Only after forecasting the current
        # fold do its outcomes become available
        # for future calibration.
        history.extend(
            current
        )

    return (
        predictions,
        alpha_schedule,
    )


def metrics(
    rows: list[dict[str, object]],
    probability_column: str,
) -> dict[str, float]:
    probabilities = [
        float(
            row[
                probability_column
            ]
        )
        for row in rows
    ]

    targets = [
        int(
            row["target"]
        )
        for row in rows
    ]

    log_losses = [
        binary_log_loss(
            target,
            probability,
        )
        for target, probability
        in zip(
            targets,
            probabilities,
            strict=True,
        )
    ]

    brier = [
        (
            probability
            - target
        ) ** 2
        for target, probability
        in zip(
            targets,
            probabilities,
            strict=True,
        )
    ]

    correct = [
        int(
            (
                probability >= 0.5
            )
            == target
        )
        for target, probability
        in zip(
            targets,
            probabilities,
            strict=True,
        )
    ]

    return {
        "log_loss":
            mean(
                log_losses
            ),
        "brier_score":
            mean(
                brier
            ),
        "accuracy":
            mean(
                correct
            ),
        "mean_probability":
            mean(
                probabilities
            ),
    }


def comparison(
    first: dict[str, float],
    second: dict[str, float],
) -> dict[str, float]:
    return {
        "log_loss_delta":
            second[
                "log_loss"
            ]
            - first[
                "log_loss"
            ],
        "brier_delta":
            second[
                "brier_score"
            ]
            - first[
                "brier_score"
            ],
        "accuracy_delta":
            second[
                "accuracy"
            ]
            - first[
                "accuracy"
            ],
    }


def main() -> None:
    rows = load_rows()

    grouped = group_by_fold(
        rows
    )

    folds = sorted(
        grouped
    )

    if len(folds) != 20:
        raise RuntimeError(
            f"Expected 20 folds, got "
            f"{len(folds)}."
        )

    (
        price_predictions,
        price_schedule,
    ) = build_prequential_predictions(
        rows,
        "mprice_probability",
    )

    (
        combined_predictions,
        combined_schedule,
    ) = build_prequential_predictions(
        rows,
        "mcombined_probability",
    )

    output_rows = []

    for row in rows:
        key = (
            int(
                row["fold"]
            ),
            str(
                row[
                    "decision_time"
                ]
            ),
        )

        (
            price_alpha,
            price_shrunk,
        ) = price_predictions[
            key
        ]

        (
            combined_alpha,
            combined_shrunk,
        ) = combined_predictions[
            key
        ]

        output_rows.append(
            {
                **row,
                "mprice_alpha":
                    price_alpha,
                "mprice_shrunk_probability":
                    price_shrunk,
                "mcombined_alpha":
                    combined_alpha,
                "mcombined_shrunk_probability":
                    combined_shrunk,
            }
        )

    all_metrics = {
        "M0":
            metrics(
                output_rows,
                "m0_probability",
            ),
        "M_PRICE_RAW":
            metrics(
                output_rows,
                "mprice_probability",
            ),
        "M_PRICE_SHRUNK":
            metrics(
                output_rows,
                "mprice_shrunk_probability",
            ),
        "M_COMBINED_RAW":
            metrics(
                output_rows,
                "mcombined_probability",
            ),
        "M_COMBINED_SHRUNK":
            metrics(
                output_rows,
                "mcombined_shrunk_probability",
            ),
    }

    # Fold 0 has no historical OOS data,
    # so alpha=1 by construction.
    calibrated_rows = [
        row
        for row in output_rows
        if int(
            row["fold"]
        ) > folds[0]
    ]

    calibrated_metrics = {
        "M0":
            metrics(
                calibrated_rows,
                "m0_probability",
            ),
        "M_PRICE_RAW":
            metrics(
                calibrated_rows,
                "mprice_probability",
            ),
        "M_PRICE_SHRUNK":
            metrics(
                calibrated_rows,
                "mprice_shrunk_probability",
            ),
        "M_COMBINED_RAW":
            metrics(
                calibrated_rows,
                "mcombined_probability",
            ),
        "M_COMBINED_SHRUNK":
            metrics(
                calibrated_rows,
                "mcombined_shrunk_probability",
            ),
    }

    comparisons = {
        "price_shrinkage_vs_raw":
            comparison(
                all_metrics[
                    "M_PRICE_RAW"
                ],
                all_metrics[
                    "M_PRICE_SHRUNK"
                ],
            ),
        "combined_shrinkage_vs_raw":
            comparison(
                all_metrics[
                    "M_COMBINED_RAW"
                ],
                all_metrics[
                    "M_COMBINED_SHRUNK"
                ],
            ),
        "shrunk_combined_vs_shrunk_price":
            comparison(
                all_metrics[
                    "M_PRICE_SHRUNK"
                ],
                all_metrics[
                    "M_COMBINED_SHRUNK"
                ],
            ),
        "shrunk_price_vs_m0":
            comparison(
                all_metrics[
                    "M0"
                ],
                all_metrics[
                    "M_PRICE_SHRUNK"
                ],
            ),
        "shrunk_combined_vs_m0":
            comparison(
                all_metrics[
                    "M0"
                ],
                all_metrics[
                    "M_COMBINED_SHRUNK"
                ],
            ),
    }

    results = {
        "experiment":
            "prequential-probability-shrinkage-4h",
        "selection_rule":
            (
                "For each fold, alpha is selected "
                "by minimum log loss using only "
                "completed OOS forecasts from "
                "strictly earlier folds."
            ),
        "shrinkage_definition":
            (
                "p_shrunk = p_m0 + "
                "alpha * (p_model - p_m0)"
            ),
        "alpha_grid":
            {
                "minimum":
                    min(
                        ALPHA_GRID
                    ),
                "maximum":
                    max(
                        ALPHA_GRID
                    ),
                "step":
                    0.01,
            },
        "cold_start_rule":
            (
                "Fold 0 uses alpha=1.0 because "
                "no prior OOS calibration sample "
                "exists."
            ),
        "oos_forecasts":
            len(
                output_rows
            ),
        "calibrated_oos_forecasts":
            len(
                calibrated_rows
            ),
        "folds":
            len(
                folds
            ),
        "all_oos_metrics":
            all_metrics,
        "post_cold_start_metrics":
            calibrated_metrics,
        "comparisons":
            comparisons,
        "price_alpha_schedule":
            price_schedule,
        "combined_alpha_schedule":
            combined_schedule,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
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

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    with ALPHA_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "fold",
                "alpha",
                "selection_observations",
                "selection_log_loss",
            ],
        )

        writer.writeheader()

        for item in price_schedule:
            writer.writerow(
                {
                    "model":
                        "M_PRICE",
                    **item,
                }
            )

        for item in combined_schedule:
            writer.writerow(
                {
                    "model":
                        "M_COMBINED",
                    **item,
                }
            )

    print(
        "PREQUENTIAL PROBABILITY SHRINKAGE"
    )
    print(
        "---------------------------------"
    )
    print(
        f"OOS forecasts: "
        f"{len(output_rows):,}"
    )
    print(
        f"Folds: {len(folds)}"
    )
    print()

    for model in (
        "M0",
        "M_PRICE_RAW",
        "M_PRICE_SHRUNK",
        "M_COMBINED_RAW",
        "M_COMBINED_SHRUNK",
    ):
        result = all_metrics[
            model
        ]

        print(
            f"{model:18s} "
            f"LL="
            f"{result['log_loss']:.8f}  "
            f"Brier="
            f"{result['brier_score']:.8f}  "
            f"Acc="
            f"{result['accuracy']:.4%}"
        )

    print()
    print(
        "SHRINKAGE EFFECTS"
    )
    print(
        "-----------------"
    )

    for name, result in (
        comparisons.items()
    ):
        print(
            f"{name}: "
            f"LL "
            f"{result['log_loss_delta']:+.8f}, "
            f"Brier "
            f"{result['brier_delta']:+.8f}, "
            f"Acc "
            f"{result['accuracy_delta']:+.4%}"
        )

    print()
    print(
        "ALPHA SCHEDULE"
    )
    print(
        "--------------"
    )

    for price_item, combined_item in zip(
        price_schedule,
        combined_schedule,
        strict=True,
    ):
        print(
            f"Fold "
            f"{price_item['fold']:2d}: "
            f"price alpha="
            f"{price_item['alpha']:.2f}, "
            f"combined alpha="
            f"{combined_item['alpha']:.2f}, "
            f"history="
            f"{price_item['selection_observations']:,}"
        )

    print()
    print(
        "Results     ->",
        RESULTS_FILE,
    )
    print(
        "Predictions ->",
        PREDICTIONS_FILE,
    )
    print(
        "Alphas      ->",
        ALPHA_FILE,
    )


if __name__ == "__main__":
    main()
