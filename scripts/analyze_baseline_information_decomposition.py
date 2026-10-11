from __future__ import annotations

import csv
import json
import math

from collections import defaultdict
from pathlib import Path
from statistics import mean


PRICE_FILE = Path(
    "data/derived/experiments/price_baseline/"
    "m0_vs_mprice_predictions.csv"
)

FLOW_FILE = Path(
    "data/derived/experiments/trade_flow_baseline/"
    "m0_vs_minfo_predictions.csv"
)

COMBINED_FILE = Path(
    "data/derived/experiments/combined_baseline/"
    "m0_vs_mcombined_predictions.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "baseline_information_decomposition"
)

SUMMARY_JSON = (
    OUTPUT_ROOT
    / "baseline_model_comparison_4h_2020_2025.json"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "baseline_model_comparison_4h_2020_2025.csv"
)

MERGED_PREDICTIONS = (
    OUTPUT_ROOT
    / "baseline_model_predictions_4h_2020_2025.csv"
)


def load_predictions(
    path: Path,
    probability_column: str,
) -> list[dict[str, object]]:
    rows = []

    with path.open(
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
            probability_column,
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
                f"{path} missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            rows.append(
                {
                    "fold":
                        int(
                            row["fold"]
                        ),
                    "decision_time":
                        row[
                            "decision_time"
                        ],
                    "target":
                        int(
                            row["target"]
                        ),
                    "m0_probability":
                        float(
                            row[
                                "m0_probability"
                            ]
                        ),
                    "model_probability":
                        float(
                            row[
                                probability_column
                            ]
                        ),
                }
            )

    if not rows:
        raise RuntimeError(
            f"No predictions in {path}"
        )

    return rows


def observation_key(
    row: dict[str, object],
) -> tuple[int, str]:
    return (
        int(
            row["fold"]
        ),
        str(
            row["decision_time"]
        ),
    )


def validate_alignment(
    reference: list[
        dict[str, object]
    ],
    candidate: list[
        dict[str, object]
    ],
    *,
    candidate_name: str,
) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"{candidate_name}: "
            "observation count differs."
        )

    for ref, other in zip(
        reference,
        candidate,
        strict=True,
    ):
        if (
            observation_key(ref)
            != observation_key(other)
        ):
            raise RuntimeError(
                f"{candidate_name}: "
                "timestamp/fold mismatch."
            )

        if (
            int(ref["target"])
            != int(other["target"])
        ):
            raise RuntimeError(
                f"{candidate_name}: "
                "target mismatch."
            )

        if not math.isclose(
            float(
                ref["m0_probability"]
            ),
            float(
                other["m0_probability"]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"{candidate_name}: "
                "M0 probability mismatch."
            )


def clip_probability(
    probability: float,
) -> float:
    epsilon = 1e-15

    return min(
        max(
            probability,
            epsilon,
        ),
        1.0 - epsilon,
    )


def log_loss(
    targets: list[int],
    probabilities: list[float],
) -> float:
    losses = []

    for target, probability in zip(
        targets,
        probabilities,
        strict=True,
    ):
        p = clip_probability(
            probability
        )

        losses.append(
            -(
                target
                * math.log(p)
                + (
                    1 - target
                )
                * math.log(
                    1.0 - p
                )
            )
        )

    return mean(
        losses
    )


def brier_score(
    targets: list[int],
    probabilities: list[float],
) -> float:
    return mean(
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
    )


def accuracy(
    targets: list[int],
    probabilities: list[float],
) -> float:
    return mean(
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
    )


def metric_bundle(
    targets: list[int],
    probabilities: list[float],
) -> dict[str, float]:
    return {
        "log_loss":
            log_loss(
                targets,
                probabilities,
            ),
        "brier_score":
            brier_score(
                targets,
                probabilities,
            ),
        "accuracy":
            accuracy(
                targets,
                probabilities,
            ),
        "mean_probability":
            mean(
                probabilities
            ),
    }


def group_by_fold(
    merged_rows: list[
        dict[str, object]
    ],
) -> dict[
    int,
    list[
        dict[str, object]
    ],
]:
    grouped = defaultdict(
        list
    )

    for row in merged_rows:
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


def model_metrics_by_fold(
    merged_rows: list[
        dict[str, object]
    ],
    probability_column: str,
) -> dict[
    int,
    dict[str, float],
]:
    output = {}

    for fold, rows in group_by_fold(
        merged_rows
    ).items():
        targets = [
            int(
                row["target"]
            )
            for row in rows
        ]

        probabilities = [
            float(
                row[
                    probability_column
                ]
            )
            for row in rows
        ]

        output[
            fold
        ] = metric_bundle(
            targets,
            probabilities,
        )

    return output


def compare_models(
    first_name: str,
    second_name: str,
    first_metrics: dict[
        str,
        float,
    ],
    second_metrics: dict[
        str,
        float,
    ],
    first_folds: dict[
        int,
        dict[str, float],
    ],
    second_folds: dict[
        int,
        dict[str, float],
    ],
) -> dict[str, object]:
    common_folds = sorted(
        set(first_folds)
        & set(second_folds)
    )

    if not common_folds:
        raise RuntimeError(
            "No common folds."
        )

    logloss_deltas = [
        (
            second_folds[
                fold
            ][
                "log_loss"
            ]
            - first_folds[
                fold
            ][
                "log_loss"
            ]
        )
        for fold in common_folds
    ]

    brier_deltas = [
        (
            second_folds[
                fold
            ][
                "brier_score"
            ]
            - first_folds[
                fold
            ][
                "brier_score"
            ]
        )
        for fold in common_folds
    ]

    accuracy_deltas = [
        (
            second_folds[
                fold
            ][
                "accuracy"
            ]
            - first_folds[
                fold
            ][
                "accuracy"
            ]
        )
        for fold in common_folds
    ]

    return {
        "first":
            first_name,
        "second":
            second_name,
        "delta_definition":
            "second_minus_first",
        "aggregate": {
            "log_loss_delta":
                second_metrics[
                    "log_loss"
                ]
                - first_metrics[
                    "log_loss"
                ],
            "brier_delta":
                second_metrics[
                    "brier_score"
                ]
                - first_metrics[
                    "brier_score"
                ],
            "accuracy_delta":
                second_metrics[
                    "accuracy"
                ]
                - first_metrics[
                    "accuracy"
                ],
        },
        "folds": {
            "count":
                len(
                    common_folds
                ),
            "second_logloss_wins":
                sum(
                    delta < 0.0
                    for delta
                    in logloss_deltas
                ),
            "second_brier_wins":
                sum(
                    delta < 0.0
                    for delta
                    in brier_deltas
                ),
            "second_accuracy_wins":
                sum(
                    delta > 0.0
                    for delta
                    in accuracy_deltas
                ),
            "mean_logloss_delta":
                mean(
                    logloss_deltas
                ),
            "mean_brier_delta":
                mean(
                    brier_deltas
                ),
            "mean_accuracy_delta":
                mean(
                    accuracy_deltas
                ),
        },
    }


def main() -> None:
    price = load_predictions(
        PRICE_FILE,
        "mprice_probability",
    )

    flow = load_predictions(
        FLOW_FILE,
        "minfo_probability",
    )

    combined = load_predictions(
        COMBINED_FILE,
        "mcombined_probability",
    )

    validate_alignment(
        price,
        flow,
        candidate_name="M_INFO",
    )

    validate_alignment(
        price,
        combined,
        candidate_name="M_COMBINED",
    )

    merged_rows = []

    for p_row, f_row, c_row in zip(
        price,
        flow,
        combined,
        strict=True,
    ):
        merged_rows.append(
            {
                "fold":
                    p_row["fold"],
                "decision_time":
                    p_row[
                        "decision_time"
                    ],
                "target":
                    p_row["target"],
                "m0_probability":
                    p_row[
                        "m0_probability"
                    ],
                "mprice_probability":
                    p_row[
                        "model_probability"
                    ],
                "minfo_probability":
                    f_row[
                        "model_probability"
                    ],
                "mcombined_probability":
                    c_row[
                        "model_probability"
                    ],
            }
        )

    targets = [
        int(
            row["target"]
        )
        for row in merged_rows
    ]

    probability_columns = {
        "M0":
            "m0_probability",
        "M_PRICE":
            "mprice_probability",
        "M_INFO":
            "minfo_probability",
        "M_COMBINED":
            "mcombined_probability",
    }

    metrics = {}

    fold_metrics = {}

    for model_name, column in (
        probability_columns.items()
    ):
        probabilities = [
            float(
                row[column]
            )
            for row in merged_rows
        ]

        metrics[
            model_name
        ] = metric_bundle(
            targets,
            probabilities,
        )

        fold_metrics[
            model_name
        ] = model_metrics_by_fold(
            merged_rows,
            column,
        )

    comparisons = [
        compare_models(
            "M0",
            "M_PRICE",
            metrics["M0"],
            metrics["M_PRICE"],
            fold_metrics["M0"],
            fold_metrics["M_PRICE"],
        ),
        compare_models(
            "M0",
            "M_INFO",
            metrics["M0"],
            metrics["M_INFO"],
            fold_metrics["M0"],
            fold_metrics["M_INFO"],
        ),
        compare_models(
            "M0",
            "M_COMBINED",
            metrics["M0"],
            metrics["M_COMBINED"],
            fold_metrics["M0"],
            fold_metrics["M_COMBINED"],
        ),
        compare_models(
            "M_PRICE",
            "M_COMBINED",
            metrics["M_PRICE"],
            metrics["M_COMBINED"],
            fold_metrics["M_PRICE"],
            fold_metrics["M_COMBINED"],
        ),
        compare_models(
            "M_INFO",
            "M_COMBINED",
            metrics["M_INFO"],
            metrics["M_COMBINED"],
            fold_metrics["M_INFO"],
            fold_metrics["M_COMBINED"],
        ),
    ]

    summary = {
        "experiment":
            "baseline-information-decomposition-4h",
        "period":
            "2020-2025",
        "oos_forecasts":
            len(
                merged_rows
            ),
        "folds":
            len(
                group_by_fold(
                    merged_rows
                )
            ),
        "first_oos_decision":
            merged_rows[
                0
            ][
                "decision_time"
            ],
        "last_oos_decision":
            merged_rows[
                -1
            ][
                "decision_time"
            ],
        "models":
            metrics,
        "comparisons":
            comparisons,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "log_loss",
                "brier_score",
                "accuracy",
                "mean_probability",
            ],
        )

        writer.writeheader()

        for model_name in (
            "M0",
            "M_PRICE",
            "M_INFO",
            "M_COMBINED",
        ):
            writer.writerow(
                {
                    "model":
                        model_name,
                    **metrics[
                        model_name
                    ],
                }
            )

    with MERGED_PREDICTIONS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                merged_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            merged_rows
        )

    print(
        "BASELINE INFORMATION DECOMPOSITION"
    )
    print(
        "----------------------------------"
    )
    print(
        f"OOS forecasts: "
        f"{len(merged_rows):,}"
    )
    print(
        f"Folds: "
        f"{len(group_by_fold(merged_rows))}"
    )
    print()

    for model_name in (
        "M0",
        "M_PRICE",
        "M_INFO",
        "M_COMBINED",
    ):
        result = metrics[
            model_name
        ]

        print(
            f"{model_name:10s} "
            f"logloss="
            f"{result['log_loss']:.8f}  "
            f"brier="
            f"{result['brier_score']:.8f}  "
            f"accuracy="
            f"{result['accuracy']:.4%}"
        )

    print()
    print(
        "PAIRWISE — SECOND MINUS FIRST"
    )
    print(
        "-----------------------------"
    )

    for comparison in comparisons:
        aggregate = comparison[
            "aggregate"
        ]

        folds = comparison[
            "folds"
        ]

        print(
            f"{comparison['first']} -> "
            f"{comparison['second']}: "
            f"LL "
            f"{aggregate['log_loss_delta']:+.8f}, "
            f"Brier "
            f"{aggregate['brier_delta']:+.8f}, "
            f"Acc "
            f"{aggregate['accuracy_delta']:+.4%}"
        )

        print(
            "  fold wins second: "
            f"LL "
            f"{folds['second_logloss_wins']}/"
            f"{folds['count']}, "
            f"Brier "
            f"{folds['second_brier_wins']}/"
            f"{folds['count']}, "
            f"Acc "
            f"{folds['second_accuracy_wins']}/"
            f"{folds['count']}"
        )

    print()
    print(
        "Summary JSON ->",
        SUMMARY_JSON,
    )
    print(
        "Summary CSV  ->",
        SUMMARY_CSV,
    )
    print(
        "Predictions  ->",
        MERGED_PREDICTIONS,
    )


if __name__ == "__main__":
    main()
