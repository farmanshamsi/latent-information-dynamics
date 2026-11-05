from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

import run_nested_latent_neural_walk_forward as base


SUPERVISED_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_24h_2020_2025.csv"
)

SUPERVISED_METADATA_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_24h_2020_2025.json"
)

# Used ONLY to preserve the original 20 OOS
# decision-time fold boundaries.
FOLD_SCHEDULE_FILE = Path(
    "data/derived/experiments/"
    "baseline_information_decomposition/"
    "baseline_model_predictions_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/robustness_24h"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "nested_latent_neural_walk_forward_24h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "nested_latent_neural_walk_forward_24h_predictions.csv"
)

SELECTION_FILE = (
    OUTPUT_ROOT
    / "nested_latent_epoch_selection_24h.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "nested_latent_refit_gradients_24h.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "nested_latent_folds_24h.csv"
)


def load_examples() -> tuple[
    list[base.Example],
    tuple[str, ...],
]:
    metadata = json.loads(
        SUPERVISED_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    if metadata[
        "horizon_hours"
    ] != 24:
        raise RuntimeError(
            "Expected 24-hour dataset."
        )

    feature_names = tuple(
        metadata[
            "features"
        ]
    )

    flow_features = tuple(
        feature
        for feature in feature_names
        if feature
        not in base.PRICE_FEATURES
    )

    if len(
        base.PRICE_FEATURES
    ) != 11:
        raise RuntimeError(
            "Expected 11 price features."
        )

    if len(
        flow_features
    ) != 20:
        raise RuntimeError(
            "Expected 20 Flow features."
        )

    examples = []

    with SUPERVISED_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for raw in reader:
            examples.append(
                base.Example(
                    decision_time=
                        base.parse_datetime(
                            raw[
                                "decision_time"
                            ]
                        ),
                    label_available_time=
                        base.parse_datetime(
                            raw[
                                "label_available_time"
                            ]
                        ),
                    target=
                        int(
                            raw[
                                "target"
                            ]
                        ),
                    price_x=
                        tuple(
                            float(
                                raw[
                                    name
                                ]
                            )
                            for name
                            in base.PRICE_FEATURES
                        ),
                    flow_x=
                        tuple(
                            float(
                                raw[
                                    name
                                ]
                            )
                            for name
                            in flow_features
                        ),
                )
            )

    if len(
        examples
    ) != metadata[
        "rows"
    ]:
        raise RuntimeError(
            "Metadata row count mismatch."
        )

    if any(
        example.label_available_time
        <= example.decision_time
        for example in examples
    ):
        raise RuntimeError(
            "Invalid label availability."
        )

    return (
        examples,
        flow_features,
    )


def load_fold_schedule() -> dict[
    int,
    list,
]:
    grouped = defaultdict(
        list
    )

    with FOLD_SCHEDULE_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for raw in reader:
            grouped[
                int(
                    raw[
                        "fold"
                    ]
                )
            ].append(
                base.parse_datetime(
                    raw[
                        "decision_time"
                    ]
                )
            )

    if set(
        grouped
    ) != set(
        range(
            20
        )
    ):
        raise RuntimeError(
            "Expected exactly 20 "
            "original folds."
        )

    for fold in range(
        20
    ):
        grouped[
            fold
        ] = sorted(
            grouped[
                fold
            ]
        )

        if not grouped[
            fold
        ]:
            raise RuntimeError(
                f"Empty fold {fold}."
            )

    return grouped


def write_csv(
    path: Path,
    rows: list[
        dict[str, object]
    ],
) -> None:
    if not rows:
        raise RuntimeError(
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


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    examples, _ = load_examples()

    example_by_time = {
        example.decision_time:
            example
        for example in examples
    }

    schedule = load_fold_schedule()

    predictions = []

    selection_rows = []

    gradient_rows = []

    fold_rows = []

    total_missing = 0

    print(
        "24H NESTED LATENT ROBUSTNESS"
    )

    print(
        "============================"
    )

    print(
        "Dataset observations:",
        f"{len(examples):,}",
    )

    print(
        "Fold schedule: frozen from "
        "primary 4h experiment"
    )

    print(
        "Target horizon: 24 hours"
    )

    print()

    for fold in range(
        20
    ):
        scheduled_times = (
            schedule[
                fold
            ]
        )

        # Keep the original outer test
        # boundary even if an end-of-sample
        # 24h target is unavailable.
        outer_test_start = (
            scheduled_times[
                0
            ]
        )

        outer_test = [
            example_by_time[
                decision_time
            ]
            for decision_time
            in scheduled_times
            if decision_time
            in example_by_time
        ]

        missing = (
            len(
                scheduled_times
            )
            - len(
                outer_test
            )
        )

        total_missing += (
            missing
        )

        if not outer_test:
            raise RuntimeError(
                f"Fold {fold} has no "
                "24h OOS observations."
            )

        outer_train = [
            example
            for example in examples
            if (
                example.decision_time
                < outer_test_start
                and
                example.label_available_time
                <= outer_test_start
            )
        ]

        if (
            len(
                outer_train
            )
            <= base.INNER_VALIDATION_OBSERVATIONS
        ):
            raise RuntimeError(
                "Outer training sample "
                "too small."
            )

        validation = (
            outer_train[
                -base.INNER_VALIDATION_OBSERVATIONS:
            ]
        )

        validation_start = (
            validation[
                0
            ].decision_time
        )

        inner_train = [
            example
            for example in outer_train
            if (
                example.decision_time
                < validation_start
                and
                example.label_available_time
                <= validation_start
            )
        ]

        if any(
            example.label_available_time
            > validation_start
            for example
            in inner_train
        ):
            raise RuntimeError(
                "Inner PIT failure."
            )

        if any(
            example.label_available_time
            > outer_test_start
            for example
            in outer_train
        ):
            raise RuntimeError(
                "Outer PIT failure."
            )

        if any(
            example.decision_time
            < outer_test_start
            for example
            in outer_test
        ):
            raise RuntimeError(
                "Invalid outer test."
            )

        seed = (
            base.BASE_SEED
            + fold
        )

        (
            price_epochs,
            price_curve,
        ) = (
            base.select_price_epochs(
                inner_train,
                validation,
                seed=seed,
                fold=fold,
            )
        )

        (
            multimodal_epochs,
            multimodal_curve,
        ) = (
            base.select_multimodal_epochs(
                inner_train,
                validation,
                seed=
                    seed
                    + 50000,
                fold=fold,
            )
        )

        selection_rows.extend(
            price_curve
        )

        selection_rows.extend(
            multimodal_curve
        )

        (
            price_probabilities,
            price_gradients,
        ) = base.refit_price(
            outer_train,
            outer_test,
            epochs=
                price_epochs,
            seed=
                seed
                + 1000000,
            fold=fold,
        )

        (
            multimodal_probabilities,
            multimodal_gradients,
        ) = base.refit_multimodal(
            outer_train,
            outer_test,
            epochs=
                multimodal_epochs,
            seed=
                seed
                + 2000000,
            fold=fold,
        )

        gradient_rows.extend(
            price_gradients
        )

        gradient_rows.extend(
            multimodal_gradients
        )

        # Historical unconditional
        # probability using only labels
        # legally available at the fold
        # boundary.
        m0_probability = mean(
            example.target
            for example in outer_train
        )

        fold_prediction_rows = []

        for (
            example,
            price_probability,
            multimodal_probability,
        ) in zip(
            outer_test,
            price_probabilities,
            multimodal_probabilities,
            strict=True,
        ):
            row = {
                "fold":
                    fold,
                "decision_time":
                    example
                    .decision_time
                    .isoformat(),
                "label_available_time":
                    example
                    .label_available_time
                    .isoformat(),
                "target":
                    example.target,
                "m0_probability":
                    m0_probability,
                "price_nn_probability":
                    price_probability,
                "multimodal_nn_probability":
                    multimodal_probability,
            }

            predictions.append(
                row
            )

            fold_prediction_rows.append(
                row
            )

        targets = [
            int(
                row[
                    "target"
                ]
            )
            for row
            in fold_prediction_rows
        ]

        m0 = base.metric_set(
            targets,
            [
                float(
                    row[
                        "m0_probability"
                    ]
                )
                for row
                in fold_prediction_rows
            ],
        )

        price_nn = base.metric_set(
            targets,
            [
                float(
                    row[
                        "price_nn_probability"
                    ]
                )
                for row
                in fold_prediction_rows
            ],
        )

        multimodal_nn = (
            base.metric_set(
                targets,
                [
                    float(
                        row[
                            "multimodal_nn_probability"
                        ]
                    )
                    for row
                    in fold_prediction_rows
                ],
            )
        )

        fold_rows.append(
            {
                "fold":
                    fold,
                "outer_test_start":
                    outer_test_start
                    .isoformat(),
                "outer_train_observations":
                    len(
                        outer_train
                    ),
                "inner_train_observations":
                    len(
                        inner_train
                    ),
                "validation_observations":
                    len(
                        validation
                    ),
                "scheduled_test_observations":
                    len(
                        scheduled_times
                    ),
                "test_observations":
                    len(
                        outer_test
                    ),
                "missing_24h_targets":
                    missing,
                "price_selected_epochs":
                    price_epochs,
                "multimodal_selected_epochs":
                    multimodal_epochs,
                "m0_log_loss":
                    m0[
                        "log_loss"
                    ],
                "price_nn_log_loss":
                    price_nn[
                        "log_loss"
                    ],
                "multimodal_nn_log_loss":
                    multimodal_nn[
                        "log_loss"
                    ],
                "multimodal_minus_price_nn":
                    (
                        multimodal_nn[
                            "log_loss"
                        ]
                        - price_nn[
                            "log_loss"
                        ]
                    ),
            }
        )

        print(
            f"Fold {fold:02d}: "
            f"Train="
            f"{len(outer_train):5d} "
            f"Test="
            f"{len(outer_test):3d} "
            f"Missing="
            f"{missing:2d} "
            f"Epochs P="
            f"{price_epochs:2d} "
            f"M="
            f"{multimodal_epochs:2d}  "
            f"M0="
            f"{m0['log_loss']:.6f} "
            f"Price="
            f"{price_nn['log_loss']:.6f} "
            f"Multi="
            f"{multimodal_nn['log_loss']:.6f}"
        )

    targets = [
        int(
            row[
                "target"
            ]
        )
        for row in predictions
    ]

    aggregate = {
        "M0":
            base.metric_set(
                targets,
                [
                    float(
                        row[
                            "m0_probability"
                        ]
                    )
                    for row
                    in predictions
                ],
            ),
        "M_PRICE_NN":
            base.metric_set(
                targets,
                [
                    float(
                        row[
                            "price_nn_probability"
                        ]
                    )
                    for row
                    in predictions
                ],
            ),
        "M_MULTIMODAL_NN":
            base.metric_set(
                targets,
                [
                    float(
                        row[
                            "multimodal_nn_probability"
                        ]
                    )
                    for row
                    in predictions
                ],
            ),
    }

    price_vs_m0 = (
        base.comparison(
            predictions,
            "m0_probability",
            "price_nn_probability",
        )
    )

    multimodal_vs_price = (
        base.comparison(
            predictions,
            "price_nn_probability",
            "multimodal_nn_probability",
        )
    )

    results = {
        "experiment":
            (
                "nested-latent-neural-"
                "walk-forward-24h-robustness"
            ),
        "status":
            (
                "Frozen robustness experiment. "
                "No architecture or optimizer "
                "retuning for the 24h target."
            ),
        "protocol": {
            "forecast_horizon_hours":
                24,
            "fold_boundaries":
                (
                    "Identical decision-time "
                    "boundaries to primary "
                    "4h experiment."
                ),
            "outer_folds":
                20,
            "outer_oos_forecasts":
                len(
                    predictions
                ),
            "missing_scheduled_targets":
                total_missing,
            "inner_validation_observations":
                base.INNER_VALIDATION_OBSERVATIONS,
            "maximum_epochs":
                base.MAX_EPOCHS,
            "batch_size":
                base.BATCH_SIZE,
            "learning_rate":
                base.LEARNING_RATE,
            "weight_decay":
                base.WEIGHT_DECAY,
            "base_seed":
                base.BASE_SEED,
            "epoch_selection":
                (
                    "Minimum chronological "
                    "inner-validation log loss."
                ),
            "refit":
                (
                    "From scratch on complete "
                    "legal outer training set."
                ),
            "label_legality":
                (
                    "label_available_time must "
                    "be <= evaluation boundary."
                ),
            "point_in_time":
                True,
            "architecture_search":
                False,
            "optimizer_search":
                False,
        },
        "metrics":
            aggregate,
        "price_nn_vs_m0":
            price_vs_m0,
        "multimodal_nn_vs_price_nn":
            multimodal_vs_price,
        "selected_epochs": {
            "price":
                [
                    row[
                        "price_selected_epochs"
                    ]
                    for row
                    in fold_rows
                ],
            "multimodal":
                [
                    row[
                        "multimodal_selected_epochs"
                    ]
                    for row
                    in fold_rows
                ],
        },
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
        GRADIENT_FILE,
        gradient_rows,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
    )

    print()
    print(
        "24H AGGREGATE RESULTS"
    )

    print(
        "---------------------"
    )

    for name in (
        "M0",
        "M_PRICE_NN",
        "M_MULTIMODAL_NN",
    ):
        item = aggregate[
            name
        ]

        print(
            f"{name:18s} "
            f"LL="
            f"{item['log_loss']:.8f}  "
            f"Brier="
            f"{item['brier_score']:.8f}  "
            f"Acc="
            f"{item['accuracy']:.4%}"
        )

    print()
    print(
        "PRICE NN - M0"
    )

    print(
        "-------------"
    )

    item = price_vs_m0[
        "log_loss"
    ]

    print(
        "LL delta:",
        f"{item['mean_difference']:+.8f}",
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

    print()
    print(
        "MULTIMODAL NN - PRICE NN"
    )

    print(
        "------------------------"
    )

    item = multimodal_vs_price[
        "log_loss"
    ]

    print(
        "LL delta:",
        f"{item['mean_difference']:+.8f}",
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

    print()
    print(
        "Selected Price epochs:",
        results[
            "selected_epochs"
        ][
            "price"
        ],
    )

    print(
        "Selected Multi epochs:",
        results[
            "selected_epochs"
        ][
            "multimodal"
        ],
    )

    print()
    print(
        "Scheduled targets missing:",
        total_missing,
    )

    print(
        "OOS forecasts:",
        len(
            predictions
        ),
    )

    print()
    print(
        "Results ->",
        RESULTS_FILE,
    )


if __name__ == "__main__":
    main()
