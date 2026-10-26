from __future__ import annotations

import csv
import json

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as clipping


RECENCY_WINDOWS = (
    None,
    1644,
    1092,
    546,
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
    "latent_neural/recency_window"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "nested_recency_window_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "nested_recency_window_predictions.csv"
)

SELECTION_FILE = (
    OUTPUT_ROOT
    / "nested_recency_window_selection.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "nested_recency_window_folds.csv"
)


def window_label(
    window: int | None,
) -> str:
    if window is None:
        return "full"

    return f"last_{window}"


def recent_subset(
    examples: list[base.Example],
    window: int | None,
) -> list[base.Example]:
    if window is None:
        return examples

    if window >= len(
        examples
    ):
        return examples

    return examples[
        -window:
    ]


def available_windows(
    observations: int,
) -> list[
    tuple[
        int | None,
        int,
    ]
]:
    """
    Full history is always included.

    Fixed windows that are identical to full history
    in an early fold are omitted to avoid duplicate
    candidates.
    """

    result = []

    seen_effective_sizes = set()

    for window in RECENCY_WINDOWS:
        effective = (
            observations
            if window is None
            else min(
                window,
                observations,
            )
        )

        if (
            effective
            in seen_effective_sizes
        ):
            continue

        seen_effective_sizes.add(
            effective
        )

        result.append(
            (
                window,
                effective,
            )
        )

    return result


def best_validation_loss(
    curve: list[
        dict[str, object]
    ],
) -> float:
    return min(
        float(
            row[
                "validation_log_loss"
            ]
        )
        for row
        in curve
    )


def augment_curve(
    curve: list[
        dict[str, object]
    ],
    *,
    model_name: str,
    window: int | None,
    observations: int,
) -> list[
    dict[str, object]
]:
    rows = []

    for row in curve:
        rows.append(
            {
                "fold":
                    row[
                        "fold"
                    ],
                "model":
                    model_name,
                "window":
                    window_label(
                        window
                    ),
                "training_observations":
                    observations,
                "epoch":
                    row[
                        "epoch"
                    ],
                "validation_log_loss":
                    row[
                        "validation_log_loss"
                    ],
            }
        )

    return rows


def select_price_recency(
    inner_train: list[base.Example],
    validation: list[base.Example],
    *,
    seed: int,
    fold: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    candidates = []

    all_rows = []

    for (
        window,
        observations,
    ) in available_windows(
        len(
            inner_train
        )
    ):
        training = recent_subset(
            inner_train,
            window,
        )

        (
            epoch,
            curve,
        ) = base.select_price_epochs(
            training,
            validation,
            seed=seed,
            fold=fold,
        )

        validation_loss = (
            best_validation_loss(
                curve
            )
        )

        candidate = {
            "window":
                window,
            "window_label":
                window_label(
                    window
                ),
            "training_observations":
                observations,
            "epoch":
                epoch,
            "validation_log_loss":
                validation_loss,
        }

        candidates.append(
            candidate
        )

        all_rows.extend(
            augment_curve(
                curve,
                model_name=
                    "PRICE_NN",
                window=
                    window,
                observations=
                    observations,
            )
        )

    selected = min(
        candidates,
        key=lambda row: (
            float(
                row[
                    "validation_log_loss"
                ]
            ),
            -int(
                row[
                    "training_observations"
                ]
            ),
        ),
    )

    full = next(
        row
        for row in candidates
        if row[
            "window"
        ] is None
    )

    return (
        selected,
        all_rows,
        full,
    )


def select_multimodal_recency(
    inner_train: list[base.Example],
    validation: list[base.Example],
    *,
    seed: int,
    fold: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    candidates = []

    all_rows = []

    for (
        window,
        observations,
    ) in available_windows(
        len(
            inner_train
        )
    ):
        training = recent_subset(
            inner_train,
            window,
        )

        (
            epoch,
            curve,
        ) = (
            base.select_multimodal_epochs(
                training,
                validation,
                seed=seed,
                fold=fold,
            )
        )

        validation_loss = (
            best_validation_loss(
                curve
            )
        )

        candidate = {
            "window":
                window,
            "window_label":
                window_label(
                    window
                ),
            "training_observations":
                observations,
            "epoch":
                epoch,
            "validation_log_loss":
                validation_loss,
        }

        candidates.append(
            candidate
        )

        all_rows.extend(
            augment_curve(
                curve,
                model_name=
                    "MULTIMODAL_NN",
                window=
                    window,
                observations=
                    observations,
            )
        )

    selected = min(
        candidates,
        key=lambda row: (
            float(
                row[
                    "validation_log_loss"
                ]
            ),
            -int(
                row[
                    "training_observations"
                ]
            ),
        ),
    )

    full = next(
        row
        for row in candidates
        if row[
            "window"
        ] is None
    )

    return (
        selected,
        all_rows,
        full,
    )


def probability_metrics(
    rows: list[
        dict[str, object]
    ],
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


def hac_summary(
    values: list[float],
) -> dict[str, object]:
    mu = mean(
        values
    )

    result = {
        "mean_difference":
            mu,
        "hac": {},
    }

    for lag in base.HAC_LAGS:
        standard_error = (
            base.hac_standard_error(
                values,
                lag,
            )
        )

        statistic = (
            mu
            / standard_error
            if standard_error > 0.0
            else 0.0
        )

        result[
            "hac"
        ][
            str(
                lag
            )
        ] = {
            "standard_error":
                standard_error,
            "t_statistic":
                statistic,
            "p_value":
                base.normal_p(
                    statistic
                ),
        }

    return result


def differential_recency_effect(
    rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """
    Difference in differences:

        (Multi recency - Multi expanding)
        -
        (Price recency - Price expanding)

    Negative values mean recency helped the multimodal
    model more than it helped the Price-only model.
    """

    log_loss_values = []

    brier_values = []

    for row in rows:
        target = int(
            row[
                "target"
            ]
        )

        price_expanding = float(
            row[
                "price_expanding_probability"
            ]
        )

        price_recency = float(
            row[
                "price_recency_probability"
            ]
        )

        multimodal_expanding = float(
            row[
                "multimodal_expanding_probability"
            ]
        )

        multimodal_recency = float(
            row[
                "multimodal_recency_probability"
            ]
        )

        price_ll_change = (
            base.observation_log_loss(
                target,
                price_recency,
            )
            - base.observation_log_loss(
                target,
                price_expanding,
            )
        )

        multimodal_ll_change = (
            base.observation_log_loss(
                target,
                multimodal_recency,
            )
            - base.observation_log_loss(
                target,
                multimodal_expanding,
            )
        )

        price_brier_change = (
            base.observation_brier(
                target,
                price_recency,
            )
            - base.observation_brier(
                target,
                price_expanding,
            )
        )

        multimodal_brier_change = (
            base.observation_brier(
                target,
                multimodal_recency,
            )
            - base.observation_brier(
                target,
                multimodal_expanding,
            )
        )

        log_loss_values.append(
            multimodal_ll_change
            - price_ll_change
        )

        brier_values.append(
            multimodal_brier_change
            - price_brier_change
        )

    return {
        "log_loss":
            hac_summary(
                log_loss_values
            ),
        "brier":
            hac_summary(
                brier_values
            ),
    }


def write_csv(
    path: Path,
    rows: list[
        dict[str, object]
    ],
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
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_comparison(
    comparison: dict[str, object],
) -> None:
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

        for lag in base.HAC_LAGS:
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


def main() -> None:
    nested_results = json.loads(
        NESTED_RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    baseline_price_epochs = [
        int(
            value
        )
        for value
        in nested_results[
            "selected_epochs"
        ][
            "price"
        ]
    ]

    baseline_multimodal_epochs = [
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

    predictions = []

    selection_rows = []

    fold_rows = []

    price_window_counts = Counter()

    multimodal_window_counts = Counter()

    print(
        "NESTED RECENCY-WINDOW WALK-FORWARD"
    )

    print(
        "=================================="
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
            frozen[0][
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

        validation = (
            outer_train[
                -base.INNER_VALIDATION_OBSERVATIONS:
            ]
        )

        validation_start = (
            validation[0]
            .decision_time
        )

        inner_train = [
            example
            for example
            in outer_train
            if (
                example.decision_time
                < validation_start
                and
                example.label_available_time
                <= validation_start
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

        price_selection_seed = (
            base.BASE_SEED
            + fold
        )

        multimodal_selection_seed = (
            base.BASE_SEED
            + fold
            + 50000
        )

        (
            selected_price,
            price_curve,
            full_price,
        ) = select_price_recency(
            inner_train,
            validation,
            seed=
                price_selection_seed,
            fold=
                fold,
        )

        (
            selected_multimodal,
            multimodal_curve,
            full_multimodal,
        ) = select_multimodal_recency(
            inner_train,
            validation,
            seed=
                multimodal_selection_seed,
            fold=
                fold,
        )

        selection_rows.extend(
            price_curve
        )

        selection_rows.extend(
            multimodal_curve
        )

        if (
            int(
                full_price[
                    "epoch"
                ]
            )
            != baseline_price_epochs[
                fold
            ]
        ):
            raise RuntimeError(
                "Price full-history epoch "
                f"reproduction failed in fold {fold}."
            )

        if (
            int(
                full_multimodal[
                    "epoch"
                ]
            )
            != baseline_multimodal_epochs[
                fold
            ]
        ):
            raise RuntimeError(
                "Multimodal full-history epoch "
                f"reproduction failed in fold {fold}."
            )

        price_window_counts[
            selected_price[
                "window_label"
            ]
        ] += 1

        multimodal_window_counts[
            selected_multimodal[
                "window_label"
            ]
        ] += 1

        price_refit_training = (
            recent_subset(
                outer_train,
                selected_price[
                    "window"
                ],
            )
        )

        multimodal_refit_training = (
            recent_subset(
                outer_train,
                selected_multimodal[
                    "window"
                ],
            )
        )

        (
            price_probabilities,
            _,
        ) = base.refit_price(
            price_refit_training,
            outer_test,
            epochs=
                int(
                    selected_price[
                        "epoch"
                    ]
                ),
            seed=
                (
                    base.BASE_SEED
                    + fold
                    + 1000000
                ),
            fold=
                fold,
        )

        (
            multimodal_probabilities,
            _,
        ) = base.refit_multimodal(
            multimodal_refit_training,
            outer_test,
            epochs=
                int(
                    selected_multimodal[
                        "epoch"
                    ]
                ),
            seed=
                (
                    base.BASE_SEED
                    + fold
                    + 2000000
                ),
            fold=
                fold,
        )

        fold_predictions = []

        for (
            index,
            example,
        ) in enumerate(
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

            if (
                previous[
                    "target"
                ]
                != example.target
            ):
                raise RuntimeError(
                    "Target mismatch."
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
                "price_expanding_probability":
                    previous[
                        "price_nn_probability"
                    ],
                "price_recency_probability":
                    price_probabilities[
                        index
                    ],
                "multimodal_expanding_probability":
                    previous[
                        "multimodal_adamw_probability"
                    ],
                "multimodal_recency_probability":
                    multimodal_probabilities[
                        index
                    ],
            }

            predictions.append(
                row
            )

            fold_predictions.append(
                row
            )

        price_expanding = (
            probability_metrics(
                fold_predictions,
                "price_expanding_probability",
            )
        )

        price_recency = (
            probability_metrics(
                fold_predictions,
                "price_recency_probability",
            )
        )

        multimodal_expanding = (
            probability_metrics(
                fold_predictions,
                "multimodal_expanding_probability",
            )
        )

        multimodal_recency = (
            probability_metrics(
                fold_predictions,
                "multimodal_recency_probability",
            )
        )

        fold_rows.append(
            {
                "fold":
                    fold,
                "price_window":
                    selected_price[
                        "window_label"
                    ],
                "price_training_observations":
                    len(
                        price_refit_training
                    ),
                "price_epochs":
                    selected_price[
                        "epoch"
                    ],
                "price_expanding_log_loss":
                    price_expanding[
                        "log_loss"
                    ],
                "price_recency_log_loss":
                    price_recency[
                        "log_loss"
                    ],
                "price_recency_minus_expanding":
                    (
                        price_recency[
                            "log_loss"
                        ]
                        - price_expanding[
                            "log_loss"
                        ]
                    ),
                "multimodal_window":
                    selected_multimodal[
                        "window_label"
                    ],
                "multimodal_training_observations":
                    len(
                        multimodal_refit_training
                    ),
                "multimodal_epochs":
                    selected_multimodal[
                        "epoch"
                    ],
                "multimodal_expanding_log_loss":
                    multimodal_expanding[
                        "log_loss"
                    ],
                "multimodal_recency_log_loss":
                    multimodal_recency[
                        "log_loss"
                    ],
                "multimodal_recency_minus_expanding":
                    (
                        multimodal_recency[
                            "log_loss"
                        ]
                        - multimodal_expanding[
                            "log_loss"
                        ]
                    ),
                "recency_multimodal_minus_recency_price":
                    (
                        multimodal_recency[
                            "log_loss"
                        ]
                        - price_recency[
                            "log_loss"
                        ]
                    ),
            }
        )

        print(
            f"Fold {fold:02d}: "
            f"Price[{selected_price['window_label']},"
            f"e={int(selected_price['epoch']):2d}] "
            f"{price_recency['log_loss']:.6f} "
            f"delta="
            f"{price_recency['log_loss'] - price_expanding['log_loss']:+.6f} "
            f"Multi[{selected_multimodal['window_label']},"
            f"e={int(selected_multimodal['epoch']):2d}] "
            f"{multimodal_recency['log_loss']:.6f} "
            f"delta="
            f"{multimodal_recency['log_loss'] - multimodal_expanding['log_loss']:+.6f}"
        )

    aggregate = {
        "M_PRICE_EXPANDING":
            probability_metrics(
                predictions,
                "price_expanding_probability",
            ),
        "M_PRICE_RECENCY":
            probability_metrics(
                predictions,
                "price_recency_probability",
            ),
        "M_MULTIMODAL_EXPANDING":
            probability_metrics(
                predictions,
                "multimodal_expanding_probability",
            ),
        "M_MULTIMODAL_RECENCY":
            probability_metrics(
                predictions,
                "multimodal_recency_probability",
            ),
    }

    price_recency_vs_expanding = (
        base.comparison(
            predictions,
            "price_expanding_probability",
            "price_recency_probability",
        )
    )

    multimodal_recency_vs_expanding = (
        base.comparison(
            predictions,
            "multimodal_expanding_probability",
            "multimodal_recency_probability",
        )
    )

    multimodal_recency_vs_price_recency = (
        base.comparison(
            predictions,
            "price_recency_probability",
            "multimodal_recency_probability",
        )
    )

    differential = (
        differential_recency_effect(
            predictions
        )
    )

    results = {
        "experiment":
            "nested-recency-window-walk-forward",
        "status":
            (
                "Developmental regime-adaptation "
                "experiment designed after observing "
                "2021-2025 Flow drift."
            ),
        "protocol": {
            "candidate_windows":
                [
                    "full",
                    1644,
                    1092,
                    546,
                ],
            "approximate_window_days":
                {
                    "546":
                        91,
                    "1092":
                        182,
                    "1644":
                        274,
                },
            "window_and_epoch_selection":
                (
                    "Jointly selected only from the "
                    "chronological inner validation block."
                ),
            "tie_break":
                "Prefer more training observations.",
            "architecture":
                "unchanged",
            "optimizer":
                "AdamW",
            "point_in_time":
                True,
        },
        "metrics":
            aggregate,
        "price_recency_vs_expanding":
            price_recency_vs_expanding,
        "multimodal_recency_vs_expanding":
            multimodal_recency_vs_expanding,
        "multimodal_recency_vs_price_recency":
            multimodal_recency_vs_price_recency,
        "differential_recency_effect":
            differential,
        "price_window_counts":
            dict(
                price_window_counts
            ),
        "multimodal_window_counts":
            dict(
                multimodal_window_counts
            ),
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
        predictions,
    )

    write_csv(
        SELECTION_FILE,
        selection_rows,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
    )

    print()
    print(
        "RECENCY-WINDOW AGGREGATE"
    )

    print(
        "------------------------"
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

    print()
    print(
        "PRICE RECENCY - EXPANDING"
    )

    print(
        "-------------------------"
    )

    print_comparison(
        price_recency_vs_expanding
    )

    print()
    print(
        "MULTIMODAL RECENCY - EXPANDING"
    )

    print(
        "------------------------------"
    )

    print_comparison(
        multimodal_recency_vs_expanding
    )

    print()
    print(
        "RECENCY MULTIMODAL - RECENCY PRICE"
    )

    print(
        "---------------------------------"
    )

    print_comparison(
        multimodal_recency_vs_price_recency
    )

    print()
    print(
        "DIFFERENTIAL RECENCY EFFECT"
    )

    print(
        "---------------------------"
    )

    print(
        "Negative means recency helps "
        "multimodal more than Price."
    )

    print_comparison(
        differential
    )

    print()
    print(
        "WINDOW SELECTION"
    )

    print(
        "----------------"
    )

    print(
        "Price:",
        dict(
            price_window_counts
        ),
    )

    print(
        "Multimodal:",
        dict(
            multimodal_window_counts
        ),
    )

    print()
    print(
        "WORST BASELINE MULTIMODAL FOLDS"
    )

    print(
        "-------------------------------"
    )

    worst = sorted(
        fold_rows,
        key=lambda row:
            (
                row[
                    "multimodal_expanding_log_loss"
                ]
                - row[
                    "price_expanding_log_loss"
                ]
            ),
        reverse=True,
    )[
        :7
    ]

    for row in worst:
        baseline_gap = (
            row[
                "multimodal_expanding_log_loss"
            ]
            - row[
                "price_expanding_log_loss"
            ]
        )

        repair = (
            -row[
                "multimodal_recency_minus_expanding"
            ]
        )

        print(
            f"Fold {int(row['fold']):02d} "
            f"baseline-gap="
            f"{baseline_gap:+.6f} "
            f"MultiWindow="
            f"{row['multimodal_window']} "
            f"epochs="
            f"{int(row['multimodal_epochs']):2d} "
            f"recency-repair="
            f"{repair:+.6f}"
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
        "Selection   ->",
        SELECTION_FILE,
    )

    print(
        "Folds       ->",
        FOLD_FILE,
    )


if __name__ == "__main__":
    main()
