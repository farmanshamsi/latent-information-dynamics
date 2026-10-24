from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as clipping


CLIP_THRESHOLDS = (
    0.10,
    0.20,
    0.40,
)

NESTED_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward"
)

NESTED_RESULTS_FILE = (
    NESTED_ROOT
    / "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/optimizer_clipping/"
    "matched_epoch_control"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "matched_epoch_clipping_control.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "matched_epoch_clipping_predictions.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "matched_epoch_clipping_folds.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "matched_epoch_clipping_gradients.csv"
)


def threshold_key(
    threshold: float,
) -> str:
    return (
        f"clip_{threshold:.2f}"
        .replace(
            ".",
            "_",
        )
    )


def probability_column(
    threshold: float,
) -> str:
    return (
        threshold_key(
            threshold
        )
        + "_probability"
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            "Cannot write empty CSV."
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def probability_metrics(
    rows: list[dict[str, object]],
    column: str,
) -> dict[str, float]:
    return base.metric_set(
        [
            int(
                row[
                    "target"
                ]
            )
            for row
            in rows
        ],
        [
            float(
                row[
                    column
                ]
            )
            for row
            in rows
        ],
    )


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    nested_results = json.loads(
        NESTED_RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    baseline_epochs = [
        int(
            value
        )
        for value
        in nested_results[
            "selected_epochs"
        ][
            "multimodal"
        ]
    ]

    if len(
        baseline_epochs
    ) != 20:
        raise RuntimeError(
            "Expected 20 baseline epoch selections."
        )

    (
        examples,
        _,
    ) = base.load_examples()

    frozen_baseline = (
        base.load_baseline_rows()
    )

    previous_predictions = (
        clipping
        .load_previous_nested_predictions()
    )

    example_by_time = {
        example.decision_time:
            example
        for example
        in examples
    }

    grouped = defaultdict(
        list
    )

    for row in frozen_baseline:
        grouped[
            int(
                row[
                    "fold"
                ]
            )
        ].append(
            row
        )

    all_predictions = []

    all_gradient_rows = []

    fold_rows = []

    reproduction_errors = []

    print(
        "MATCHED-EPOCH GRADIENT-CLIPPING CONTROL"
    )

    print(
        "======================================="
    )

    for fold in range(
        20
    ):
        frozen = sorted(
            grouped[
                fold
            ],
            key=lambda row:
                row[
                    "decision_time"
                ],
        )

        outer_test_start = (
            frozen[
                0
            ][
                "decision_time"
            ]
        )

        outer_train = [
            example
            for example
            in examples
            if (
                example.decision_time
                < outer_test_start
                and
                example.label_available_time
                <= outer_test_start
            )
        ]

        outer_test = [
            example_by_time[
                row[
                    "decision_time"
                ]
            ]
            for row
            in frozen
        ]

        epochs = (
            baseline_epochs[
                fold
            ]
        )

        refit_seed = (
            base.BASE_SEED
            + fold
            + 2000000
        )

        (
            reproduced_probabilities,
            _,
        ) = base.refit_multimodal(
            outer_train,
            outer_test,
            epochs=
                epochs,
            seed=
                refit_seed,
            fold=
                fold,
        )

        stored_probabilities = []

        for example in outer_test:
            previous = (
                previous_predictions[
                    (
                        fold,
                        example.decision_time,
                    )
                ]
            )

            stored_probabilities.append(
                float(
                    previous[
                        "multimodal_adamw_probability"
                    ]
                )
            )

        reproduction_error = float(
            np.max(
                np.abs(
                    np.asarray(
                        reproduced_probabilities,
                        dtype=np.float64,
                    )
                    - np.asarray(
                        stored_probabilities,
                        dtype=np.float64,
                    )
                )
            )
        )

        reproduction_errors.append(
            reproduction_error
        )

        if (
            reproduction_error
            > 1e-10
        ):
            raise RuntimeError(
                "Unclipped AdamW reproduction failure "
                f"in fold {fold}: "
                f"{reproduction_error}"
            )

        threshold_predictions = {}

        threshold_gradients = {}

        for threshold in (
            CLIP_THRESHOLDS
        ):
            (
                probabilities,
                gradients,
            ) = (
                clipping.refit_threshold(
                    outer_train,
                    outer_test,
                    threshold=
                        threshold,
                    epochs=
                        epochs,
                    seed=
                        refit_seed,
                    fold=
                        fold,
                )
            )

            threshold_predictions[
                threshold
            ] = probabilities

            threshold_gradients[
                threshold
            ] = gradients

            all_gradient_rows.extend(
                gradients
            )

        fold_predictions = []

        for index, example in enumerate(
            outer_test
        ):
            previous = (
                previous_predictions[
                    (
                        fold,
                        example.decision_time,
                    )
                ]
            )

            row = {
                "fold":
                    fold,
                "decision_time":
                    example
                    .decision_time
                    .isoformat(),
                "target":
                    example.target,
                "price_nn_probability":
                    previous[
                        "price_nn_probability"
                    ],
                "adamw_probability":
                    reproduced_probabilities[
                        index
                    ],
            }

            for threshold in (
                CLIP_THRESHOLDS
            ):
                row[
                    probability_column(
                        threshold
                    )
                ] = (
                    threshold_predictions[
                        threshold
                    ][
                        index
                    ]
                )

            fold_predictions.append(
                row
            )

            all_predictions.append(
                row
            )

        adamw_metrics = (
            probability_metrics(
                fold_predictions,
                "adamw_probability",
            )
        )

        price_metrics = (
            probability_metrics(
                fold_predictions,
                "price_nn_probability",
            )
        )

        fold_row = {
            "fold":
                fold,
            "epochs":
                epochs,
            "reproduction_max_abs_error":
                reproduction_error,
            "price_nn_log_loss":
                price_metrics[
                    "log_loss"
                ],
            "adamw_log_loss":
                adamw_metrics[
                    "log_loss"
                ],
        }

        display = []

        for threshold in (
            CLIP_THRESHOLDS
        ):
            key = threshold_key(
                threshold
            )

            metrics = (
                probability_metrics(
                    fold_predictions,
                    probability_column(
                        threshold
                    ),
                )
            )

            control = (
                clipping
                .gradient_control_summary(
                    threshold_gradients[
                        threshold
                    ]
                )
            )

            delta = (
                metrics[
                    "log_loss"
                ]
                - adamw_metrics[
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_log_loss"
            ] = (
                metrics[
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_minus_adamw"
            ] = delta

            fold_row[
                f"{key}_clipped_fraction"
            ] = (
                control[
                    "clipped_fraction"
                ]
            )

            display.append(
                f"C{threshold:.2f}="
                f"{metrics['log_loss']:.6f}"
                f"({delta:+.6f})"
            )

        fold_rows.append(
            fold_row
        )

        print(
            f"Fold {fold:02d}: "
            f"epochs={epochs:2d} "
            f"AdamW="
            f"{adamw_metrics['log_loss']:.6f} "
            + " ".join(
                display
            )
        )

    aggregate = {
        "M_PRICE_NN":
            probability_metrics(
                all_predictions,
                "price_nn_probability",
            ),
        "M_MULTIMODAL_ADAMW":
            probability_metrics(
                all_predictions,
                "adamw_probability",
            ),
    }

    comparisons = {}

    price_comparisons = {}

    gradient_summaries = {}

    for threshold in (
        CLIP_THRESHOLDS
    ):
        key = threshold_key(
            threshold
        )

        aggregate[
            key
        ] = probability_metrics(
            all_predictions,
            probability_column(
                threshold
            ),
        )

        comparisons[
            key
        ] = base.comparison(
            all_predictions,
            "adamw_probability",
            probability_column(
                threshold
            ),
        )

        price_comparisons[
            key
        ] = base.comparison(
            all_predictions,
            "price_nn_probability",
            probability_column(
                threshold
            ),
        )

        gradient_summaries[
            key
        ] = (
            clipping
            .gradient_control_summary(
                [
                    row
                    for row
                    in all_gradient_rows
                    if float(
                        row[
                            "max_norm"
                        ]
                    )
                    == threshold
                ]
            )
        )

    results = {
        "experiment":
            "matched-epoch-gradient-clipping-control",
        "purpose":
            (
                "Isolate clipping from epoch-selection effects "
                "using the exact epoch budget selected by the "
                "original unclipped AdamW model."
            ),
        "protocol": {
            "outer_folds":
                20,
            "oos_forecasts":
                len(
                    all_predictions
                ),
            "clip_thresholds":
                list(
                    CLIP_THRESHOLDS
                ),
            "epoch_budget":
                (
                    "Original multimodal AdamW "
                    "inner-validation-selected epoch "
                    "for each outer fold."
                ),
            "same_initialization":
                True,
            "same_minibatch_order":
                True,
            "same_training_data":
                True,
            "same_standardization":
                True,
            "only_intervention":
                "global gradient clipping",
            "developmental":
                True,
        },
        "baseline_reproduction": {
            "maximum_absolute_probability_error":
                max(
                    reproduction_errors
                )
        },
        "metrics":
            aggregate,
        "clipping_vs_adamw":
            comparisons,
        "clipping_vs_price_nn":
            price_comparisons,
        "gradient_control":
            gradient_summaries,
        "folds":
            fold_rows,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        PREDICTIONS_FILE,
        all_predictions,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
    )

    write_csv(
        GRADIENT_FILE,
        all_gradient_rows,
    )

    print()
    print(
        "BASELINE REPRODUCTION"
    )

    print(
        "---------------------"
    )

    print(
        "Maximum absolute probability error:",
        f"{max(reproduction_errors):.12g}",
    )

    print()
    print(
        "MATCHED-EPOCH AGGREGATE RESULTS"
    )

    print(
        "-------------------------------"
    )

    for name, metrics in (
        aggregate.items()
    ):
        print(
            f"{name:28s} "
            f"LL="
            f"{metrics['log_loss']:.8f} "
            f"Brier="
            f"{metrics['brier_score']:.8f} "
            f"Acc="
            f"{metrics['accuracy']:.4%}"
        )

    for threshold in (
        CLIP_THRESHOLDS
    ):
        key = threshold_key(
            threshold
        )

        comparison = (
            comparisons[
                key
            ]
        )

        gradient = (
            gradient_summaries[
                key
            ]
        )

        print()
        print(
            f"MATCHED CLIP {threshold:.2f} - ADAMW"
        )

        print(
            "-" * 31
        )

        for metric in (
            "log_loss",
            "brier",
        ):
            item = comparison[
                metric
            ]

            print(
                f"{metric:8s} delta="
                f"{item['mean_difference']:+.8f}"
            )

            for lag in (
                base.HAC_LAGS
            ):
                hac = item[
                    "hac"
                ][
                    str(
                        lag
                    )
                ]

                print(
                    f"  lag {lag:2d}: "
                    f"t="
                    f"{hac['t_statistic']:+.4f} "
                    f"p="
                    f"{hac['p_value']:.6f}"
                )

        print(
            "Clip rate:",
            f"{gradient['clipped_fraction']:.2%}",
        )

        print(
            "Raw p99:",
            f"{gradient['raw_p99']:.6f}",
            "Effective p99:",
            f"{gradient['effective_p99']:.6f}",
        )

    print()
    print(
        "FAILURE-FOLD MATCHED REPAIR"
    )

    print(
        "---------------------------"
    )

    worst = sorted(
        fold_rows,
        key=lambda row:
            (
                float(
                    row[
                        "adamw_log_loss"
                    ]
                )
                - float(
                    row[
                        "price_nn_log_loss"
                    ]
                )
            ),
        reverse=True,
    )[
        :7
    ]

    for row in worst:
        text = (
            f"Fold {int(row['fold']):02d} "
            f"gap="
            f"{row['adamw_log_loss'] - row['price_nn_log_loss']:+.6f}"
        )

        for threshold in (
            CLIP_THRESHOLDS
        ):
            key = threshold_key(
                threshold
            )

            text += (
                f" C{threshold:.2f}="
                f"{row[f'{key}_minus_adamw']:+.6f}"
            )

        print(
            text
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
        "Folds       ->",
        FOLD_FILE,
    )

    print(
        "Gradients   ->",
        GRADIENT_FILE,
    )


if __name__ == "__main__":
    main()
