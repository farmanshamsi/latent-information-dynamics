from __future__ import annotations

import csv
import json

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn

import run_nested_latent_neural_walk_forward as base

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    capture_gradient_snapshot,
    set_research_seed,
)
from latent_information_dynamics.models.latent.gradient_control import (
    clip_gradients_by_global_norm,
)


CLIP_THRESHOLDS = (
    0.10,
    0.20,
    0.40,
)

NESTED_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward"
)

BASELINE_PREDICTIONS_FILE = (
    NESTED_ROOT
    / "nested_latent_neural_walk_forward_predictions.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/optimizer_clipping"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "nested_adamw_clipping_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "nested_adamw_clipping_predictions.csv"
)

SELECTION_FILE = (
    OUTPUT_ROOT
    / "nested_adamw_clipping_selection.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "nested_adamw_clipping_gradients.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "nested_adamw_clipping_folds.csv"
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


def load_previous_nested_predictions() -> dict[
    tuple[int, object],
    dict[str, object],
]:
    rows = {}

    with BASELINE_PREDICTIONS_FILE.open(
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
            "linear_price_probability",
            "price_nn_probability",
            "multimodal_nn_probability",
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
                "Previous nested predictions missing "
                f"columns: {sorted(missing)}"
            )

        for raw in reader:
            fold = int(
                raw[
                    "fold"
                ]
            )

            decision_time = (
                base.parse_datetime(
                    raw[
                        "decision_time"
                    ]
                )
            )

            key = (
                fold,
                decision_time,
            )

            if key in rows:
                raise RuntimeError(
                    "Duplicate nested prediction key."
                )

            rows[
                key
            ] = {
                "target":
                    int(
                        raw[
                            "target"
                        ]
                    ),
                "m0_probability":
                    float(
                        raw[
                            "m0_probability"
                        ]
                    ),
                "linear_price_probability":
                    float(
                        raw[
                            "linear_price_probability"
                        ]
                    ),
                "price_nn_probability":
                    float(
                        raw[
                            "price_nn_probability"
                        ]
                    ),
                "multimodal_adamw_probability":
                    float(
                        raw[
                            "multimodal_nn_probability"
                        ]
                    ),
            }

    if len(
        rows
    ) != 10840:
        raise RuntimeError(
            "Expected 10,840 nested predictions, "
            f"received {len(rows)}."
        )

    return rows


def train_clipped_epoch(
    *,
    model: MultimodalLatentModel,
    optimizer: torch.optim.Optimizer,
    price_x: torch.Tensor,
    flow_x: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    max_norm: float,
    fold: int,
    epoch: int,
    record_gradients: bool,
) -> list[dict[str, object]]:
    permutation = torch.randperm(
        price_x.shape[0],
        generator=generator,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    rows = []

    for batch, start in enumerate(
        range(
            0,
            price_x.shape[0],
            base.BATCH_SIZE,
        )
    ):
        indices = permutation[
            start:
            start + base.BATCH_SIZE
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            price_x[
                indices
            ],
            flow_x[
                indices
            ],
        )

        loss = criterion(
            logits,
            target[
                indices
            ],
        )

        loss.backward()

        raw = None

        if record_gradients:
            raw = (
                capture_gradient_snapshot(
                    model
                )
            )

        control = (
            clip_gradients_by_global_norm(
                model.parameters(),
                max_norm=max_norm,
            )
        )

        if record_gradients:
            effective = (
                capture_gradient_snapshot(
                    model
                )
            )

            rows.append(
                {
                    "fold":
                        fold,
                    "max_norm":
                        max_norm,
                    "epoch":
                        epoch,
                    "batch":
                        batch,
                    "batch_observations":
                        int(
                            indices.shape[
                                0
                            ]
                        ),
                    "loss":
                        float(
                            loss
                            .detach()
                            .item()
                        ),
                    "raw_total_norm":
                        raw.total_norm,
                    "effective_total_norm":
                        effective.total_norm,
                    "gradient_scale":
                        control.scale,
                    "gradient_clipped":
                        int(
                            control.clipped
                        ),
                    "raw_price_encoder_norm":
                        raw.price_encoder_norm,
                    "effective_price_encoder_norm":
                        effective.price_encoder_norm,
                    "raw_flow_encoder_norm":
                        raw.flow_encoder_norm,
                    "effective_flow_encoder_norm":
                        effective.flow_encoder_norm,
                    "raw_fusion_norm":
                        raw.fusion_norm,
                    "effective_fusion_norm":
                        effective.fusion_norm,
                    "raw_prediction_head_norm":
                        raw.prediction_head_norm,
                    "effective_prediction_head_norm":
                        effective.prediction_head_norm,
                    "raw_max_abs_gradient":
                        raw.max_abs_gradient,
                    "effective_max_abs_gradient":
                        effective.max_abs_gradient,
                    "nonfinite_gradient_elements":
                        raw.nonfinite_gradient_elements,
                }
            )

        optimizer.step()

    return rows


def select_epoch_for_threshold(
    inner_train: list[base.Example],
    validation: list[base.Example],
    *,
    threshold: float,
    seed: int,
    fold: int,
) -> tuple[
    int,
    float,
    list[dict[str, object]],
]:
    (
        price_train,
        flow_train,
        target_train,
        price_validation,
        flow_validation,
    ) = base.make_multimodal_tensors(
        inner_train,
        validation,
    )

    validation_targets = [
        example.target
        for example
        in validation
    ]

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

    # Every clipping threshold sees exactly the same
    # minibatch order.
    generator.manual_seed(
        seed + 200000
    )

    best_epoch = 1

    best_loss = float(
        "inf"
    )

    rows = []

    for epoch in range(
        1,
        base.MAX_EPOCHS + 1,
    ):
        model.train()

        train_clipped_epoch(
            model=model,
            optimizer=optimizer,
            price_x=price_train,
            flow_x=flow_train,
            target=target_train,
            generator=generator,
            max_norm=threshold,
            fold=fold,
            epoch=epoch,
            record_gradients=False,
        )

        model.eval()

        with torch.no_grad():
            probabilities = (
                model.predict_proba(
                    price_validation,
                    flow_validation,
                )
            )

        validation_loss = (
            base.validation_log_loss(
                validation_targets,
                probabilities,
            )
        )

        rows.append(
            {
                "fold":
                    fold,
                "max_norm":
                    threshold,
                "epoch":
                    epoch,
                "validation_log_loss":
                    validation_loss,
            }
        )

        if (
            validation_loss
            < best_loss
            - 1e-12
        ):
            best_loss = (
                validation_loss
            )

            best_epoch = (
                epoch
            )

    return (
        best_epoch,
        best_loss,
        rows,
    )


def refit_threshold(
    outer_train: list[base.Example],
    outer_test: list[base.Example],
    *,
    threshold: float,
    epochs: int,
    seed: int,
    fold: int,
) -> tuple[
    list[float],
    list[dict[str, object]],
]:
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

    # Same refit minibatch sequence as the original
    # nested multimodal AdamW experiment.
    generator.manual_seed(
        seed + 400000
    )

    gradient_rows = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        gradient_rows.extend(
            train_clipped_epoch(
                model=model,
                optimizer=optimizer,
                price_x=price_train,
                flow_x=flow_train,
                target=target_train,
                generator=generator,
                max_norm=threshold,
                fold=fold,
                epoch=epoch,
                record_gradients=True,
            )
        )

    model.eval()

    with torch.no_grad():
        probabilities = (
            model.predict_proba(
                price_test,
                flow_test,
            )
        )

    return (
        [
            float(
                value
            )
            for value
            in probabilities
            .cpu()
            .tolist()
        ],
        gradient_rows,
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


def quantile(
    values: list[float],
    q: float,
) -> float:
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def gradient_control_summary(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError(
            "Gradient rows cannot be empty."
        )

    raw = [
        float(
            row[
                "raw_total_norm"
            ]
        )
        for row
        in rows
    ]

    effective = [
        float(
            row[
                "effective_total_norm"
            ]
        )
        for row
        in rows
    ]

    scales = [
        float(
            row[
                "gradient_scale"
            ]
        )
        for row
        in rows
    ]

    clipped = [
        int(
            row[
                "gradient_clipped"
            ]
        )
        for row
        in rows
    ]

    return {
        "steps":
            len(
                rows
            ),
        "clipped_steps":
            sum(
                clipped
            ),
        "clipped_fraction":
            mean(
                clipped
            ),
        "mean_gradient_scale":
            mean(
                scales
            ),
        "raw_mean":
            mean(
                raw
            ),
        "raw_p95":
            quantile(
                raw,
                0.95,
            ),
        "raw_p99":
            quantile(
                raw,
                0.99,
            ),
        "raw_max":
            max(
                raw
            ),
        "effective_mean":
            mean(
                effective
            ),
        "effective_p95":
            quantile(
                effective,
                0.95,
            ),
        "effective_p99":
            quantile(
                effective,
                0.99,
            ),
        "effective_max":
            max(
                effective
            ),
    }


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
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    (
        examples,
        _,
    ) = base.load_examples()

    frozen_baseline = (
        base.load_baseline_rows()
    )

    previous_predictions = (
        load_previous_nested_predictions()
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

    all_selection_rows = []

    all_gradient_rows = []

    fold_rows = []

    adaptive_thresholds = []

    print(
        "NESTED ADAMW GRADIENT-CLIPPING EXPERIMENT"
    )

    print(
        "========================================="
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

        selection_seed = (
            base.BASE_SEED
            + fold
            + 50000
        )

        refit_seed = (
            base.BASE_SEED
            + fold
            + 2000000
        )

        threshold_selection = {}

        threshold_predictions = {}

        threshold_gradients = {}

        for threshold in sorted(
            CLIP_THRESHOLDS,
            reverse=True,
        ):
            (
                selected_epoch,
                validation_loss,
                selection_rows,
            ) = select_epoch_for_threshold(
                inner_train,
                validation,
                threshold=
                    threshold,
                seed=
                    selection_seed,
                fold=
                    fold,
            )

            all_selection_rows.extend(
                selection_rows
            )

            (
                probabilities,
                gradient_rows,
            ) = refit_threshold(
                outer_train,
                outer_test,
                threshold=
                    threshold,
                epochs=
                    selected_epoch,
                seed=
                    refit_seed,
                fold=
                    fold,
            )

            all_gradient_rows.extend(
                gradient_rows
            )

            threshold_selection[
                threshold
            ] = {
                "epoch":
                    selected_epoch,
                "validation_log_loss":
                    validation_loss,
            }

            threshold_predictions[
                threshold
            ] = probabilities

            threshold_gradients[
                threshold
            ] = gradient_rows

        # Because thresholds were iterated from least
        # aggressive to most aggressive, strict improvement
        # below means ties prefer less clipping.
        adaptive_threshold = min(
            threshold_selection,
            key=lambda threshold: (
                threshold_selection[
                    threshold
                ][
                    "validation_log_loss"
                ],
                -threshold,
            ),
        )

        adaptive_thresholds.append(
            adaptive_threshold
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

            if (
                previous[
                    "target"
                ]
                != example.target
            ):
                raise RuntimeError(
                    "Target mismatch against previous "
                    "nested experiment."
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
                "m0_probability":
                    previous[
                        "m0_probability"
                    ],
                "linear_price_probability":
                    previous[
                        "linear_price_probability"
                    ],
                "price_nn_probability":
                    previous[
                        "price_nn_probability"
                    ],
                "multimodal_adamw_probability":
                    previous[
                        "multimodal_adamw_probability"
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

            row[
                "adaptive_clipped_probability"
            ] = (
                threshold_predictions[
                    adaptive_threshold
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
                "multimodal_adamw_probability",
            )
        )

        price_metrics = (
            probability_metrics(
                fold_predictions,
                "price_nn_probability",
            )
        )

        fixed_metrics = {
            threshold:
                probability_metrics(
                    fold_predictions,
                    probability_column(
                        threshold
                    ),
                )
            for threshold
            in CLIP_THRESHOLDS
        }

        adaptive_metrics = (
            probability_metrics(
                fold_predictions,
                "adaptive_clipped_probability",
            )
        )

        adaptive_control = (
            gradient_control_summary(
                threshold_gradients[
                    adaptive_threshold
                ]
            )
        )

        fold_row = {
            "fold":
                fold,
            "adaptive_threshold":
                adaptive_threshold,
            "adaptive_epochs":
                threshold_selection[
                    adaptive_threshold
                ][
                    "epoch"
                ],
            "adaptive_validation_log_loss":
                threshold_selection[
                    adaptive_threshold
                ][
                    "validation_log_loss"
                ],
            "price_nn_log_loss":
                price_metrics[
                    "log_loss"
                ],
            "adamw_log_loss":
                adamw_metrics[
                    "log_loss"
                ],
            "adaptive_log_loss":
                adaptive_metrics[
                    "log_loss"
                ],
            "adaptive_minus_adamw":
                (
                    adaptive_metrics[
                        "log_loss"
                    ]
                    - adamw_metrics[
                        "log_loss"
                    ]
                ),
            "adaptive_minus_price_nn":
                (
                    adaptive_metrics[
                        "log_loss"
                    ]
                    - price_metrics[
                        "log_loss"
                    ]
                ),
            "adaptive_clipped_fraction":
                adaptive_control[
                    "clipped_fraction"
                ],
        }

        for threshold in (
            CLIP_THRESHOLDS
        ):
            key = threshold_key(
                threshold
            )

            control = (
                gradient_control_summary(
                    threshold_gradients[
                        threshold
                    ]
                )
            )

            fold_row[
                f"{key}_epochs"
            ] = (
                threshold_selection[
                    threshold
                ][
                    "epoch"
                ]
            )

            fold_row[
                f"{key}_validation_log_loss"
            ] = (
                threshold_selection[
                    threshold
                ][
                    "validation_log_loss"
                ]
            )

            fold_row[
                f"{key}_outer_log_loss"
            ] = (
                fixed_metrics[
                    threshold
                ][
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_minus_adamw"
            ] = (
                fixed_metrics[
                    threshold
                ][
                    "log_loss"
                ]
                - adamw_metrics[
                    "log_loss"
                ]
            )

            fold_row[
                f"{key}_clipped_fraction"
            ] = (
                control[
                    "clipped_fraction"
                ]
            )

        fold_rows.append(
            fold_row
        )

        fixed_display = " ".join(
            (
                f"C{threshold:.2f}="
                f"{fixed_metrics[threshold]['log_loss']:.6f}"
            )
            for threshold
            in CLIP_THRESHOLDS
        )

        print(
            f"Fold {fold:02d}: "
            f"AdamW="
            f"{adamw_metrics['log_loss']:.6f} "
            f"{fixed_display} "
            f"Adaptive="
            f"{adaptive_metrics['log_loss']:.6f} "
            f"[c={adaptive_threshold:.2f}, "
            f"e="
            f"{threshold_selection[adaptive_threshold]['epoch']:2d}]"
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
                "multimodal_adamw_probability",
            ),
    }

    for threshold in (
        CLIP_THRESHOLDS
    ):
        aggregate[
            threshold_key(
                threshold
            )
        ] = probability_metrics(
            all_predictions,
            probability_column(
                threshold
            ),
        )

    aggregate[
        "M_ADAPTIVE_CLIPPED"
    ] = probability_metrics(
        all_predictions,
        "adaptive_clipped_probability",
    )

    fixed_comparisons = {}

    fixed_gradient_summaries = {}

    for threshold in (
        CLIP_THRESHOLDS
    ):
        key = threshold_key(
            threshold
        )

        fixed_comparisons[
            key
        ] = base.comparison(
            all_predictions,
            "multimodal_adamw_probability",
            probability_column(
                threshold
            ),
        )

        fixed_gradient_summaries[
            key
        ] = gradient_control_summary(
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

    adaptive_vs_adamw = (
        base.comparison(
            all_predictions,
            "multimodal_adamw_probability",
            "adaptive_clipped_probability",
        )
    )

    adaptive_vs_price = (
        base.comparison(
            all_predictions,
            "price_nn_probability",
            "adaptive_clipped_probability",
        )
    )

    adaptive_gradient_rows = []

    adaptive_by_fold = {
        int(
            row[
                "fold"
            ]
        ):
            float(
                row[
                    "adaptive_threshold"
                ]
            )
        for row
        in fold_rows
    }

    for row in all_gradient_rows:
        fold = int(
            row[
                "fold"
            ]
        )

        threshold = float(
            row[
                "max_norm"
            ]
        )

        if (
            threshold
            == adaptive_by_fold[
                fold
            ]
        ):
            adaptive_gradient_rows.append(
                row
            )

    adaptive_gradient_summary = (
        gradient_control_summary(
            adaptive_gradient_rows
        )
    )

    threshold_counts = Counter(
        adaptive_thresholds
    )

    adaptive_fold_wins = sum(
        float(
            row[
                "adaptive_minus_adamw"
            ]
        )
        < 0.0
        for row
        in fold_rows
    )

    results = {
        "experiment":
            "nested-adamw-gradient-clipping-4h",
        "status":
            (
                "Developmental experiment. "
                "Clipping thresholds were specified "
                "after analysis of 2021-2025 gradient "
                "diagnostics and require future "
                "holdout confirmation."
            ),
        "protocol": {
            "outer_folds":
                20,
            "outer_oos_forecasts":
                len(
                    all_predictions
                ),
            "clip_thresholds":
                list(
                    CLIP_THRESHOLDS
                ),
            "fixed_threshold_epoch_selection":
                (
                    "Epoch selected independently for "
                    "each threshold using chronological "
                    "inner validation."
                ),
            "adaptive_strategy":
                (
                    "Threshold selected by minimum "
                    "inner-validation log loss after "
                    "within-threshold epoch selection."
                ),
            "architecture":
                "unchanged",
            "optimizer":
                "AdamW",
            "learning_rate":
                base.LEARNING_RATE,
            "weight_decay":
                base.WEIGHT_DECAY,
            "batch_size":
                base.BATCH_SIZE,
            "maximum_epochs":
                base.MAX_EPOCHS,
            "point_in_time":
                True,
            "paired_model_initialization":
                True,
            "paired_minibatch_order":
                True,
        },
        "metrics":
            aggregate,
        "fixed_threshold_vs_adamw":
            fixed_comparisons,
        "adaptive_vs_adamw":
            adaptive_vs_adamw,
        "adaptive_vs_price_nn":
            adaptive_vs_price,
        "adaptive_threshold_counts": {
            str(
                threshold
            ):
                threshold_counts[
                    threshold
                ]
            for threshold
            in CLIP_THRESHOLDS
        },
        "adaptive_fold_wins_vs_adamw":
            adaptive_fold_wins,
        "fixed_gradient_control":
            fixed_gradient_summaries,
        "adaptive_gradient_control":
            adaptive_gradient_summary,
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
        SELECTION_FILE,
        all_selection_rows,
    )

    write_csv(
        GRADIENT_FILE,
        all_gradient_rows,
    )

    write_csv(
        FOLD_FILE,
        fold_rows,
    )

    print()
    print(
        "AGGREGATE OPTIMIZER RESULTS"
    )

    print(
        "---------------------------"
    )

    for name, item in (
        aggregate.items()
    ):
        print(
            f"{name:28s} "
            f"LL="
            f"{item['log_loss']:.8f} "
            f"Brier="
            f"{item['brier_score']:.8f} "
            f"Acc="
            f"{item['accuracy']:.4%}"
        )

    for threshold in (
        CLIP_THRESHOLDS
    ):
        key = threshold_key(
            threshold
        )

        print()
        print(
            f"CLIP {threshold:.2f} - ADAMW"
        )

        print(
            "-" * 24
        )

        print_comparison(
            fixed_comparisons[
                key
            ]
        )

        gradient_summary = (
            fixed_gradient_summaries[
                key
            ]
        )

        print(
            "Clip rate: "
            f"{gradient_summary['clipped_fraction']:.2%}"
        )

        print(
            "Raw p99: "
            f"{gradient_summary['raw_p99']:.6f} "
            "Effective p99: "
            f"{gradient_summary['effective_p99']:.6f}"
        )

    print()
    print(
        "ADAPTIVE CLIPPING - ADAMW"
    )

    print(
        "-------------------------"
    )

    print_comparison(
        adaptive_vs_adamw
    )

    print()
    print(
        "ADAPTIVE CLIPPING - PRICE NN"
    )

    print(
        "----------------------------"
    )

    print_comparison(
        adaptive_vs_price
    )

    print()
    print(
        "ADAPTIVE SELECTION"
    )

    print(
        "------------------"
    )

    print(
        "Threshold counts:",
        dict(
            sorted(
                threshold_counts.items()
            )
        ),
    )

    print(
        "Fold wins vs AdamW:",
        f"{adaptive_fold_wins}/20",
    )

    print(
        "Adaptive clip rate:",
        f"{adaptive_gradient_summary['clipped_fraction']:.2%}",
    )

    print(
        "Adaptive raw gradient: "
        f"mean="
        f"{adaptive_gradient_summary['raw_mean']:.6f} "
        f"p95="
        f"{adaptive_gradient_summary['raw_p95']:.6f} "
        f"p99="
        f"{adaptive_gradient_summary['raw_p99']:.6f} "
        f"max="
        f"{adaptive_gradient_summary['raw_max']:.6f}"
    )

    print(
        "Adaptive effective gradient: "
        f"mean="
        f"{adaptive_gradient_summary['effective_mean']:.6f} "
        f"p95="
        f"{adaptive_gradient_summary['effective_p95']:.6f} "
        f"p99="
        f"{adaptive_gradient_summary['effective_p99']:.6f} "
        f"max="
        f"{adaptive_gradient_summary['effective_max']:.6f}"
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
        :5
    ]

    for row in worst:
        fold = int(
            row[
                "fold"
            ]
        )

        baseline_delta = (
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
        )

        print(
            f"Fold {fold:02d} "
            f"AdamW-Price="
            f"{baseline_delta:+.6f} "
            f"Adaptive-AdamW="
            f"{float(row['adaptive_minus_adamw']):+.6f} "
            f"clip="
            f"{float(row['adaptive_threshold']):.2f} "
            f"rate="
            f"{float(row['adaptive_clipped_fraction']):.1%}"
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
        "Gradients   ->",
        GRADIENT_FILE,
    )

    print(
        "Folds       ->",
        FOLD_FILE,
    )


if __name__ == "__main__":
    main()
