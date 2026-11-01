from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as clipping

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
)


NESTED_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward"
)

NESTED_RESULTS_FILE = (
    NESTED_ROOT
    / "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

DRIFT_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/regime_drift/"
    "multimodal_regime_drift_folds.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/modality_intervention"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "multimodal_modality_intervention.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "multimodal_modality_intervention_predictions.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "multimodal_modality_intervention_folds.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "multimodal_modality_intervention_correlations.csv"
)


def probability_metrics(
    rows: list[dict[str, object]],
    column: str,
) -> dict[str, float]:
    return base.metric_set(
        [
            int(row["target"])
            for row in rows
        ],
        [
            float(row[column])
            for row in rows
        ],
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
        writer.writerows(rows)


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
                str(lag)
            ]

            print(
                f"  lag {lag:2d}: "
                f"t={hac['t_statistic']:+.4f} "
                f"p={hac['p_value']:.6f}"
            )


def main() -> None:
    torch.set_num_threads(1)

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

    if len(selected_epochs) != 20:
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

    grouped = defaultdict(list)

    for row in frozen_baseline:
        grouped[
            int(row["fold"])
        ].append(row)

    predictions = []
    fold_rows = []
    reproduction_errors = []

    print(
        "MULTIMODAL MODALITY INTERVENTION"
    )

    print(
        "================================"
    )

    for fold in range(20):
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

        epochs = (
            selected_epochs[
                fold
            ]
        )

        seed = (
            base.BASE_SEED
            + fold
            + 2000000
        )

        set_research_seed(seed)

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

        for epoch in range(
            1,
            epochs + 1,
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

        model.eval()

        zero_price = (
            torch.zeros_like(
                price_test
            )
        )

        zero_flow = (
            torch.zeros_like(
                flow_test
            )
        )

        with torch.no_grad():
            full_probability = (
                model.predict_proba(
                    price_test,
                    flow_test,
                )
            )

            flow_neutral_probability = (
                model.predict_proba(
                    price_test,
                    zero_flow,
                )
            )

            price_neutral_probability = (
                model.predict_proba(
                    zero_price,
                    flow_test,
                )
            )

            both_neutral_probability = (
                model.predict_proba(
                    zero_price,
                    zero_flow,
                )
            )

        full_probability = np.asarray(
            full_probability
            .cpu()
            .tolist(),
            dtype=np.float64,
        )

        flow_neutral_probability = np.asarray(
            flow_neutral_probability
            .cpu()
            .tolist(),
            dtype=np.float64,
        )

        price_neutral_probability = np.asarray(
            price_neutral_probability
            .cpu()
            .tolist(),
            dtype=np.float64,
        )

        both_neutral_probability = np.asarray(
            both_neutral_probability
            .cpu()
            .tolist(),
            dtype=np.float64,
        )

        stored = np.asarray(
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

        reproduction_error = float(
            np.max(
                np.abs(
                    full_probability
                    - stored
                )
            )
        )

        reproduction_errors.append(
            reproduction_error
        )

        if reproduction_error > 1e-10:
            raise RuntimeError(
                "Baseline reproduction failure "
                f"in fold {fold}: "
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
                "full_probability":
                    full_probability[
                        index
                    ],
                "flow_neutral_probability":
                    flow_neutral_probability[
                        index
                    ],
                "price_neutral_probability":
                    price_neutral_probability[
                        index
                    ],
                "both_neutral_probability":
                    both_neutral_probability[
                        index
                    ],
            }

            fold_predictions.append(row)
            predictions.append(row)

        metrics = {
            "price_nn":
                probability_metrics(
                    fold_predictions,
                    "price_nn_probability",
                ),
            "full":
                probability_metrics(
                    fold_predictions,
                    "full_probability",
                ),
            "flow_neutral":
                probability_metrics(
                    fold_predictions,
                    "flow_neutral_probability",
                ),
            "price_neutral":
                probability_metrics(
                    fold_predictions,
                    "price_neutral_probability",
                ),
            "both_neutral":
                probability_metrics(
                    fold_predictions,
                    "both_neutral_probability",
                ),
        }

        fold_rows.append(
            {
                "fold":
                    fold,
                "epochs":
                    epochs,
                "price_nn_log_loss":
                    metrics[
                        "price_nn"
                    ][
                        "log_loss"
                    ],
                "full_log_loss":
                    metrics[
                        "full"
                    ][
                        "log_loss"
                    ],
                "flow_neutral_log_loss":
                    metrics[
                        "flow_neutral"
                    ][
                        "log_loss"
                    ],
                "price_neutral_log_loss":
                    metrics[
                        "price_neutral"
                    ][
                        "log_loss"
                    ],
                "both_neutral_log_loss":
                    metrics[
                        "both_neutral"
                    ][
                        "log_loss"
                    ],
                "flow_neutral_minus_full":
                    (
                        metrics[
                            "flow_neutral"
                        ][
                            "log_loss"
                        ]
                        - metrics[
                            "full"
                        ][
                            "log_loss"
                        ]
                    ),
                "price_neutral_minus_full":
                    (
                        metrics[
                            "price_neutral"
                        ][
                            "log_loss"
                        ]
                        - metrics[
                            "full"
                        ][
                            "log_loss"
                        ]
                    ),
                "full_minus_price_nn":
                    (
                        metrics[
                            "full"
                        ][
                            "log_loss"
                        ]
                        - metrics[
                            "price_nn"
                        ][
                            "log_loss"
                        ]
                    ),
                "flow_neutral_minus_price_nn":
                    (
                        metrics[
                            "flow_neutral"
                        ][
                            "log_loss"
                        ]
                        - metrics[
                            "price_nn"
                        ][
                            "log_loss"
                        ]
                    ),
                "reproduction_max_abs_error":
                    reproduction_error,
            }
        )

        print(
            f"Fold {fold:02d}: "
            f"e={epochs:2d} "
            f"Full="
            f"{metrics['full']['log_loss']:.6f} "
            f"Flow0="
            f"{metrics['flow_neutral']['log_loss']:.6f} "
            f"delta="
            f"{metrics['flow_neutral']['log_loss'] - metrics['full']['log_loss']:+.6f} "
            f"Price0="
            f"{metrics['price_neutral']['log_loss']:.6f}"
        )

    aggregate = {
        "M_PRICE_NN":
            probability_metrics(
                predictions,
                "price_nn_probability",
            ),
        "M_MULTIMODAL_FULL":
            probability_metrics(
                predictions,
                "full_probability",
            ),
        "M_FLOW_NEUTRAL":
            probability_metrics(
                predictions,
                "flow_neutral_probability",
            ),
        "M_PRICE_NEUTRAL":
            probability_metrics(
                predictions,
                "price_neutral_probability",
            ),
        "M_BOTH_NEUTRAL":
            probability_metrics(
                predictions,
                "both_neutral_probability",
            ),
    }

    flow_neutral_vs_full = (
        base.comparison(
            predictions,
            "full_probability",
            "flow_neutral_probability",
        )
    )

    price_neutral_vs_full = (
        base.comparison(
            predictions,
            "full_probability",
            "price_neutral_probability",
        )
    )

    flow_neutral_vs_price = (
        base.comparison(
            predictions,
            "price_nn_probability",
            "flow_neutral_probability",
        )
    )

    correlations = []

    if DRIFT_FILE.exists():
        drift = pd.read_csv(
            DRIFT_FILE
        )

        intervention = pd.DataFrame(
            fold_rows
        )

        merged = intervention.merge(
            drift,
            on="fold",
            how="inner",
            validate="one_to_one",
        )

        repair = (
            -merged[
                "flow_neutral_minus_full"
            ].to_numpy(
                dtype=np.float64
            )
        )

        drift_columns = [
            column
            for column
            in merged.columns
            if (
                column.startswith(
                    "flow_"
                )
                and column
                not in {
                    "flow_neutral_log_loss",
                    "flow_neutral_minus_full",
                    "flow_neutral_minus_price_nn",
                }
            )
        ]

        for column in drift_columns:
            values = merged[
                column
            ].to_numpy(
                dtype=np.float64
            )

            correlations.append(
                {
                    "diagnostic":
                        column,
                    "target":
                        "flow_neutralization_repair",
                    "pearson":
                        correlation(
                            values,
                            repair,
                            method="pearson",
                        ),
                    "spearman":
                        correlation(
                            values,
                            repair,
                            method="spearman",
                        ),
                }
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        PREDICTIONS_FILE,
        predictions,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
    )

    if correlations:
        write_csv(
            CORRELATION_FILE,
            correlations,
        )

    results = {
        "experiment":
            "multimodal-modality-intervention",
        "status":
            (
                "Post-hoc developmental mechanistic "
                "ablation. Neutralized modalities are "
                "replaced by their outer-training mean "
                "in standardized feature space."
            ),
        "baseline_reproduction": {
            "maximum_absolute_probability_error":
                max(
                    reproduction_errors
                )
        },
        "metrics":
            aggregate,
        "flow_neutral_vs_full":
            flow_neutral_vs_full,
        "price_neutral_vs_full":
            price_neutral_vs_full,
        "flow_neutral_vs_price_nn":
            flow_neutral_vs_price,
        "drift_repair_correlations":
            correlations,
        "folds":
            fold_rows,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
        "MODALITY INTERVENTION AGGREGATE"
    )

    print(
        "-------------------------------"
    )

    for name, metrics in aggregate.items():
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
        "FLOW NEUTRAL - FULL MULTIMODAL"
    )

    print(
        "------------------------------"
    )

    print_comparison(
        flow_neutral_vs_full
    )

    print()
    print(
        "PRICE NEUTRAL - FULL MULTIMODAL"
    )

    print(
        "-------------------------------"
    )

    print_comparison(
        price_neutral_vs_full
    )

    print()
    print(
        "FLOW NEUTRAL - PRICE NN"
    )

    print(
        "-----------------------"
    )

    print_comparison(
        flow_neutral_vs_price
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
                "full_minus_price_nn"
            ],
        reverse=True,
    )[:7]

    for row in worst:
        repair = (
            -row[
                "flow_neutral_minus_full"
            ]
        )

        print(
            f"Fold {int(row['fold']):02d} "
            f"gap="
            f"{row['full_minus_price_nn']:+.6f} "
            f"Flow0-repair="
            f"{repair:+.6f} "
            f"Flow0-vs-Price="
            f"{row['flow_neutral_minus_price_nn']:+.6f}"
        )

    if correlations:
        print()
        print(
            "FLOW DRIFT VS FLOW-NEUTRALIZATION REPAIR"
        )

        print(
            "----------------------------------------"
        )

        ranked = sorted(
            correlations,
            key=lambda row:
                abs(
                    row[
                        "spearman"
                    ]
                ),
            reverse=True,
        )

        for row in ranked:
            print(
                f"{row['diagnostic']:34s} "
                f"Pearson="
                f"{row['pearson']:+.3f} "
                f"Spearman="
                f"{row['spearman']:+.3f}"
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

    if correlations:
        print(
            "Correlations ->",
            CORRELATION_FILE,
        )


if __name__ == "__main__":
    main()
