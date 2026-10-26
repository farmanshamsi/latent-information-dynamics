from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as clipping

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)


FIXED_EPOCH_BUDGETS = (
    1,
    2,
    5,
    10,
    20,
    50,
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
    "latent_neural/training_horizon"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "multimodal_training_horizon_ablation.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "multimodal_training_horizon_predictions.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "multimodal_training_horizon_folds.csv"
)


def epoch_key(
    epoch: int,
) -> str:
    return (
        f"epoch_{epoch:02d}"
    )


def probability_column(
    epoch: int,
) -> str:
    return (
        epoch_key(
            epoch
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
                rows[0].keys()
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
                row["target"]
            )
            for row
            in rows
        ],
        [
            float(
                row[column]
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

    selected_epochs = [
        int(value)
        for value
        in nested_results[
            "selected_epochs"
        ][
            "multimodal"
        ]
    ]

    if len(
        selected_epochs
    ) != 20:
        raise RuntimeError(
            "Expected 20 selected epochs."
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
                row["fold"]
            )
        ].append(
            row
        )

    all_predictions = []
    fold_rows = []
    reproduction_errors = []

    print(
        "MULTIMODAL TRAINING-HORIZON ABLATION"
    )

    print(
        "===================================="
    )

    for fold in range(
        20
    ):
        frozen = sorted(
            grouped[fold],
            key=lambda row:
                row["decision_time"],
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

        outer_test = [
            example_by_time[
                row["decision_time"]
            ]
            for row
            in frozen
        ]

        (
            price_train,
            flow_train,
            target_train,
            price_test,
            flow_test,
        ) = base.make_multimodal_tensors(
            outer_train,
            outer_test,
        )

        selected_epoch = (
            selected_epochs[
                fold
            ]
        )

        checkpoints = set(
            FIXED_EPOCH_BUDGETS
        )

        checkpoints.add(
            selected_epoch
        )

        max_epoch = max(
            checkpoints
        )

        seed = (
            base.BASE_SEED
            + fold
            + 2000000
        )

        set_research_seed(
            seed
        )

        model = (
            MultimodalLatentModel()
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=
                    base.LEARNING_RATE,
                weight_decay=
                    base.WEIGHT_DECAY,
            )
        )

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            seed + 400000
        )

        checkpoint_probabilities = {}

        for epoch in range(
            1,
            max_epoch + 1,
        ):
            model.train()

            base.train_multimodal_epoch(
                model=model,
                optimizer=optimizer,
                price_x=price_train,
                flow_x=flow_train,
                target=target_train,
                generator=generator,
                record_gradients=False,
                fold=fold,
                epoch=epoch,
            )

            if epoch not in checkpoints:
                continue

            model.eval()

            with torch.no_grad():
                probabilities = (
                    model.predict_proba(
                        price_test,
                        flow_test,
                    )
                )

            checkpoint_probabilities[
                epoch
            ] = np.asarray(
                probabilities
                .cpu()
                .tolist(),
                dtype=np.float64,
            )

        stored_selected = np.asarray(
            [
                previous_predictions[
                    (
                        fold,
                        example.decision_time,
                    )
                ][
                    "multimodal_adamw_probability"
                ]
                for example
                in outer_test
            ],
            dtype=np.float64,
        )

        reproduced_selected = (
            checkpoint_probabilities[
                selected_epoch
            ]
        )

        reproduction_error = float(
            np.max(
                np.abs(
                    reproduced_selected
                    - stored_selected
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
                "Nested AdamW reproduction "
                f"failed in fold {fold}: "
                f"{reproduction_error}"
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
                "selected_adamw_probability":
                    stored_selected[
                        index
                    ],
            }

            for epoch in (
                FIXED_EPOCH_BUDGETS
            ):
                row[
                    probability_column(
                        epoch
                    )
                ] = (
                    checkpoint_probabilities[
                        epoch
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

        price_metrics = (
            probability_metrics(
                fold_predictions,
                "price_nn_probability",
            )
        )

        selected_metrics = (
            probability_metrics(
                fold_predictions,
                "selected_adamw_probability",
            )
        )

        fold_row = {
            "fold":
                fold,
            "selected_epoch":
                selected_epoch,
            "price_nn_log_loss":
                price_metrics[
                    "log_loss"
                ],
            "selected_adamw_log_loss":
                selected_metrics[
                    "log_loss"
                ],
            "selected_minus_price_nn":
                (
                    selected_metrics[
                        "log_loss"
                    ]
                    - price_metrics[
                        "log_loss"
                    ]
                ),
            "reproduction_max_abs_error":
                reproduction_error,
        }

        display = []

        for epoch in (
            FIXED_EPOCH_BUDGETS
        ):
            metrics = (
                probability_metrics(
                    fold_predictions,
                    probability_column(
                        epoch
                    ),
                )
            )

            key = epoch_key(
                epoch
            )

            fold_row[
                f"{key}_log_loss"
            ] = (
                metrics[
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_minus_selected"
            ] = (
                metrics[
                    "log_loss"
                ]
                - selected_metrics[
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_minus_price"
            ] = (
                metrics[
                    "log_loss"
                ]
                - price_metrics[
                    "log_loss"
                ]
            )

            display.append(
                f"E{epoch:02d}="
                f"{metrics['log_loss']:.6f}"
            )

        fold_rows.append(
            fold_row
        )

        print(
            f"Fold {fold:02d}: "
            f"selected={selected_epoch:2d} "
            f"Price="
            f"{price_metrics['log_loss']:.6f} "
            f"Selected="
            f"{selected_metrics['log_loss']:.6f} "
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
        "M_MULTIMODAL_SELECTED":
            probability_metrics(
                all_predictions,
                "selected_adamw_probability",
            ),
    }

    comparisons_vs_selected = {}

    comparisons_vs_price = {}

    for epoch in (
        FIXED_EPOCH_BUDGETS
    ):
        key = epoch_key(
            epoch
        )

        aggregate[
            key
        ] = probability_metrics(
            all_predictions,
            probability_column(
                epoch
            ),
        )

        comparisons_vs_selected[
            key
        ] = base.comparison(
            all_predictions,
            "selected_adamw_probability",
            probability_column(
                epoch
            ),
        )

        comparisons_vs_price[
            key
        ] = base.comparison(
            all_predictions,
            "price_nn_probability",
            probability_column(
                epoch
            ),
        )

    selected_epoch_array = np.asarray(
        [
            row[
                "selected_epoch"
            ]
            for row
            in fold_rows
        ],
        dtype=np.float64,
    )

    oos_gap_array = np.asarray(
        [
            row[
                "selected_minus_price_nn"
            ]
            for row
            in fold_rows
        ],
        dtype=np.float64,
    )

    selected_epoch_pearson = float(
        np.corrcoef(
            selected_epoch_array,
            oos_gap_array,
        )[
            0,
            1,
        ]
    )

    # Average-rank Spearman without an additional
    # dependency.
    epoch_ranks = (
        np.asarray(
            __import__(
                "pandas"
            )
            .Series(
                selected_epoch_array
            )
            .rank(
                method="average"
            ),
            dtype=np.float64,
        )
    )

    gap_ranks = (
        np.asarray(
            __import__(
                "pandas"
            )
            .Series(
                oos_gap_array
            )
            .rank(
                method="average"
            ),
            dtype=np.float64,
        )
    )

    selected_epoch_spearman = float(
        np.corrcoef(
            epoch_ranks,
            gap_ranks,
        )[
            0,
            1,
        ]
    )

    # Ex-post diagnostic only. Never use this field
    # for model selection or future deployment.
    for row in fold_rows:
        candidates = {
            epoch:
                float(
                    row[
                        f"{epoch_key(epoch)}_log_loss"
                    ]
                )
            for epoch
            in FIXED_EPOCH_BUDGETS
        }

        best_epoch = min(
            candidates,
            key=candidates.get,
        )

        row[
            "ex_post_best_fixed_epoch"
        ] = best_epoch

        row[
            "ex_post_best_fixed_log_loss"
        ] = candidates[
            best_epoch
        ]

    results = {
        "experiment":
            "multimodal-training-horizon-ablation",
        "status":
            (
                "Developmental post-hoc diagnostic. "
                "Fixed epoch budgets are evaluated on "
                "already-observed 2021-2025 outer folds "
                "and must not be treated as final "
                "hyperparameter selection."
            ),
        "protocol": {
            "fixed_epoch_budgets":
                list(
                    FIXED_EPOCH_BUDGETS
                ),
            "outer_folds":
                20,
            "same_initialization":
                True,
            "same_minibatch_order":
                True,
            "same_optimizer":
                "AdamW",
            "same_learning_rate":
                base.LEARNING_RATE,
            "same_weight_decay":
                base.WEIGHT_DECAY,
            "same_architecture":
                True,
            "point_in_time_training":
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
        "fixed_epoch_vs_selected":
            comparisons_vs_selected,
        "fixed_epoch_vs_price_nn":
            comparisons_vs_price,
        "selected_epoch_oos_gap_correlation": {
            "pearson":
                selected_epoch_pearson,
            "spearman":
                selected_epoch_spearman,
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
        all_predictions,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
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
        "TRAINING-HORIZON AGGREGATE"
    )

    print(
        "--------------------------"
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
        "FIXED EPOCH - SELECTED ADAMW"
    )

    print(
        "----------------------------"
    )

    for epoch in (
        FIXED_EPOCH_BUDGETS
    ):
        key = epoch_key(
            epoch
        )

        comparison = (
            comparisons_vs_selected[
                key
            ][
                "log_loss"
            ]
        )

        print(
            f"epoch={epoch:2d} "
            f"LL delta="
            f"{comparison['mean_difference']:+.8f}"
        )

        print(
            "  HAC p:",
            " ".join(
                (
                    f"{lag}:"
                    f"{comparison['hac'][str(lag)]['p_value']:.4f}"
                )
                for lag
                in base.HAC_LAGS
            ),
        )

    print()
    print(
        "FIXED EPOCH - PRICE NN"
    )

    print(
        "----------------------"
    )

    for epoch in (
        FIXED_EPOCH_BUDGETS
    ):
        key = epoch_key(
            epoch
        )

        comparison = (
            comparisons_vs_price[
                key
            ][
                "log_loss"
            ]
        )

        print(
            f"epoch={epoch:2d} "
            f"LL delta="
            f"{comparison['mean_difference']:+.8f}"
        )

    print()
    print(
        "SELECTED EPOCH / OOS FAILURE"
    )

    print(
        "----------------------------"
    )

    print(
        "Pearson:",
        f"{selected_epoch_pearson:+.3f}",
    )

    print(
        "Spearman:",
        f"{selected_epoch_spearman:+.3f}",
    )

    print()
    print(
        "WORST MULTIMODAL FOLDS"
    )

    print(
        "----------------------"
    )

    worst = sorted(
        fold_rows,
        key=lambda row:
            row[
                "selected_minus_price_nn"
            ],
        reverse=True,
    )[
        :7
    ]

    for row in worst:
        text = (
            f"Fold {int(row['fold']):02d} "
            f"selected="
            f"{int(row['selected_epoch']):2d} "
            f"gap="
            f"{row['selected_minus_price_nn']:+.6f} "
            f"expost-best="
            f"{int(row['ex_post_best_fixed_epoch']):2d}"
        )

        for epoch in (
            FIXED_EPOCH_BUDGETS
        ):
            text += (
                f" E{epoch:02d}="
                f"{row[f'{epoch_key(epoch)}_minus_price']:+.6f}"
            )

        print(
            text
        )

    print()
    print(
        "EX-POST BEST EPOCH COUNTS"
    )

    print(
        "-------------------------"
    )

    counts = {
        epoch: 0
        for epoch
        in FIXED_EPOCH_BUDGETS
    }

    for row in fold_rows:
        counts[
            int(
                row[
                    "ex_post_best_fixed_epoch"
                ]
            )
        ] += 1

    print(
        counts
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


if __name__ == "__main__":
    main()
