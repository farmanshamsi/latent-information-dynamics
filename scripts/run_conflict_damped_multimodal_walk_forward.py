from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as previous

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.conflict_control import (
    apply_shared_conflict_damping,
)
from latent_information_dynamics.models.latent.gradient_interference import (
    capture_shared_gradient_interference,
)


NESTED_RESULTS_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward/"
    "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/conflict_damping"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "conflict_damped_multimodal_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "conflict_damped_multimodal_predictions.csv"
)

CONTROL_FILE = (
    OUTPUT_ROOT
    / "conflict_damped_multimodal_control.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "conflict_damped_multimodal_folds.csv"
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


def metrics(
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


def train_model(
    *,
    outer_train,
    outer_test,
    epochs: int,
    seed: int,
    fold: int,
    damp_conflict: bool,
):
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
            lr=base.LEARNING_RATE,
            weight_decay=
                base.WEIGHT_DECAY,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 400000
    )

    control_rows = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        permutation = torch.randperm(
            price_train.shape[0],
            generator=generator,
        )

        for batch, start in enumerate(
            range(
                0,
                price_train.shape[0],
                base.BATCH_SIZE,
            )
        ):
            indices = permutation[
                start:
                start
                + base.BATCH_SIZE
            ]

            batch_price = (
                price_train[
                    indices
                ]
            )

            batch_flow = (
                flow_train[
                    indices
                ]
            )

            batch_target = (
                target_train[
                    indices
                ]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_price,
                batch_flow,
            )

            loss = criterion(
                logits,
                batch_target,
            )

            loss.backward()

            interference = (
                capture_shared_gradient_interference(
                    model,
                    batch_price,
                    batch_flow,
                    batch_target,
                )
            )

            if damp_conflict:
                control = (
                    apply_shared_conflict_damping(
                        model,
                        cosine=
                            interference
                            .price_flow_cosine,
                    )
                )

                scale = control.scale
                applied = control.applied

            else:
                scale = 1.0
                applied = False

            control_rows.append(
                {
                    "fold":
                        fold,
                    "model":
                        (
                            "damped"
                            if damp_conflict
                            else "instrumented_control"
                        ),
                    "epoch":
                        epoch,
                    "batch":
                        batch,
                    "loss":
                        float(
                            loss
                            .detach()
                            .item()
                        ),
                    "price_flow_cosine":
                        interference
                        .price_flow_cosine,
                    "shared_gradient_scale":
                        scale,
                    "damping_applied":
                        int(
                            applied
                        ),
                }
            )

            optimizer.step()

    model.eval()

    with torch.no_grad():
        probabilities = (
            model.predict_proba(
                price_test,
                flow_test,
            )
            .cpu()
            .numpy()
        )

    return (
        probabilities,
        control_rows,
    )


def control_summary(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
    scales = np.asarray(
        [
            float(
                row[
                    "shared_gradient_scale"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    applied = np.asarray(
        [
            int(
                row[
                    "damping_applied"
                ]
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    cosines = np.asarray(
        [
            float(
                row[
                    "price_flow_cosine"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(
        cosines
    )

    cosines = cosines[
        finite
    ]

    return {
        "steps":
            len(rows),
        "damped_steps":
            int(
                np.sum(
                    applied
                )
            ),
        "damped_fraction":
            float(
                np.mean(
                    applied
                )
            ),
        "mean_scale":
            float(
                np.mean(
                    scales
                )
            ),
        "minimum_scale":
            float(
                np.min(
                    scales
                )
            ),
        "mean_cosine":
            float(
                np.mean(
                    cosines
                )
            ),
        "negative_cosine_fraction":
            float(
                np.mean(
                    cosines < 0.0
                )
            ),
    }


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

    (
        examples,
        _,
    ) = base.load_examples()

    frozen = base.load_baseline_rows()

    stored = (
        previous
        .load_previous_nested_predictions()
    )

    example_by_time = {
        example.decision_time:
            example
        for example
        in examples
    }

    grouped = defaultdict(list)

    for row in frozen:
        grouped[
            int(row["fold"])
        ].append(row)

    all_predictions = []
    all_control_rows = []
    fold_rows = []

    reproduction_errors = []

    print(
        "CONFLICT-DAMPED MULTIMODAL WALK-FORWARD"
    )

    print(
        "======================================="
    )

    for fold in range(20):
        baseline_rows = sorted(
            grouped[fold],
            key=lambda row:
                row["decision_time"],
        )

        test_start = baseline_rows[
            0
        ][
            "decision_time"
        ]

        outer_train = [
            example
            for example in examples
            if (
                example.decision_time
                < test_start
                and
                example.label_available_time
                <= test_start
            )
        ]

        outer_test = [
            example_by_time[
                row[
                    "decision_time"
                ]
            ]
            for row in baseline_rows
        ]

        epochs = selected_epochs[
            fold
        ]

        seed = (
            base.BASE_SEED
            + fold
            + 2000000
        )

        (
            control_probability,
            control_rows,
        ) = train_model(
            outer_train=outer_train,
            outer_test=outer_test,
            epochs=epochs,
            seed=seed,
            fold=fold,
            damp_conflict=False,
        )

        (
            damped_probability,
            damped_rows,
        ) = train_model(
            outer_train=outer_train,
            outer_test=outer_test,
            epochs=epochs,
            seed=seed,
            fold=fold,
            damp_conflict=True,
        )

        all_control_rows.extend(
            damped_rows
        )

        stored_probability = np.asarray(
            [
                stored[
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
                    control_probability
                    - stored_probability
                )
            )
        )

        reproduction_errors.append(
            reproduction_error
        )

        if reproduction_error > 1e-10:
            raise RuntimeError(
                "Instrumented control failed "
                f"reproduction in fold {fold}: "
                f"{reproduction_error}"
            )

        fold_predictions = []

        for index, example in enumerate(
            outer_test
        ):
            previous_row = stored[
                (
                    fold,
                    example.decision_time,
                )
            ]

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
                    previous_row[
                        "price_nn_probability"
                    ],
                "adamw_probability":
                    control_probability[
                        index
                    ],
                "conflict_damped_probability":
                    damped_probability[
                        index
                    ],
            }

            fold_predictions.append(
                row
            )

            all_predictions.append(
                row
            )

        adamw = metrics(
            fold_predictions,
            "adamw_probability",
        )

        damped = metrics(
            fold_predictions,
            "conflict_damped_probability",
        )

        price = metrics(
            fold_predictions,
            "price_nn_probability",
        )

        summary = control_summary(
            damped_rows
        )

        fold_rows.append(
            {
                "fold":
                    fold,
                "epochs":
                    epochs,
                "adamw_log_loss":
                    adamw[
                        "log_loss"
                    ],
                "damped_log_loss":
                    damped[
                        "log_loss"
                    ],
                "price_nn_log_loss":
                    price[
                        "log_loss"
                    ],
                "damped_minus_adamw":
                    (
                        damped[
                            "log_loss"
                        ]
                        - adamw[
                            "log_loss"
                        ]
                    ),
                "damped_minus_price":
                    (
                        damped[
                            "log_loss"
                        ]
                        - price[
                            "log_loss"
                        ]
                    ),
                "damped_fraction":
                    summary[
                        "damped_fraction"
                    ],
                "mean_scale":
                    summary[
                        "mean_scale"
                    ],
                "minimum_scale":
                    summary[
                        "minimum_scale"
                    ],
            }
        )

        print(
            f"Fold {fold:02d}: "
            f"e={epochs:2d} "
            f"AdamW="
            f"{adamw['log_loss']:.6f} "
            f"Damped="
            f"{damped['log_loss']:.6f} "
            f"delta="
            f"{damped['log_loss'] - adamw['log_loss']:+.6f} "
            f"active="
            f"{summary['damped_fraction']:.1%} "
            f"scale="
            f"{summary['mean_scale']:.3f}"
        )

    aggregate = {
        "M_PRICE_NN":
            metrics(
                all_predictions,
                "price_nn_probability",
            ),
        "M_MULTIMODAL_ADAMW":
            metrics(
                all_predictions,
                "adamw_probability",
            ),
        "M_CONFLICT_DAMPED":
            metrics(
                all_predictions,
                "conflict_damped_probability",
            ),
    }

    damped_vs_adamw = (
        base.comparison(
            all_predictions,
            "adamw_probability",
            "conflict_damped_probability",
        )
    )

    damped_vs_price = (
        base.comparison(
            all_predictions,
            "price_nn_probability",
            "conflict_damped_probability",
        )
    )

    overall_control = (
        control_summary(
            all_control_rows
        )
    )

    results = {
        "experiment":
            "matched-conflict-damped-multimodal",
        "status":
            (
                "Developmental causal intervention. "
                "Conflict diagnostic and damping rule "
                "were designed after observing "
                "2021-2025 results."
            ),
        "protocol": {
            "architecture":
                "unchanged",
            "optimizer":
                "AdamW",
            "epoch_budget":
                (
                    "Exactly matched to original "
                    "nested multimodal AdamW."
                ),
            "initialization":
                "matched",
            "minibatch_order":
                "matched",
            "intervention":
                (
                    "When Price-context and Flow-context "
                    "shared-gradient cosine is negative, "
                    "multiply the existing full fusion/head "
                    "gradient by 1 + cosine."
                ),
            "encoder_gradients_modified":
                False,
            "shared_gradient_parameters":
                "fusion + prediction head",
            "adamw_weight_decay_unchanged":
                True,
            "point_in_time":
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
        "damped_vs_adamw":
            damped_vs_adamw,
        "damped_vs_price_nn":
            damped_vs_price,
        "control":
            overall_control,
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
        CONTROL_FILE,
        all_control_rows,
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
        "CONFLICT-DAMPING AGGREGATE"
    )

    print(
        "--------------------------"
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

    for comparison, title in (
        (
            damped_vs_adamw,
            "CONFLICT DAMPED - ADAMW",
        ),
        (
            damped_vs_price,
            "CONFLICT DAMPED - PRICE NN",
        ),
    ):
        print()
        print(
            title
        )

        print(
            "-" * len(title)
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

            for lag in base.HAC_LAGS:
                hac = item[
                    "hac"
                ][
                    str(lag)
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
        "CONTROL ACTIVITY"
    )

    print(
        "----------------"
    )

    print(
        "Damped steps:",
        f"{overall_control['damped_fraction']:.2%}",
    )

    print(
        "Mean shared-gradient scale:",
        f"{overall_control['mean_scale']:.4f}",
    )

    print(
        "Minimum shared-gradient scale:",
        f"{overall_control['minimum_scale']:.4f}",
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
                    "adamw_log_loss"
                ]
                - row[
                    "price_nn_log_loss"
                ]
            ),
        reverse=True,
    )[
        :8
    ]

    for row in worst:
        baseline_gap = (
            row[
                "adamw_log_loss"
            ]
            - row[
                "price_nn_log_loss"
            ]
        )

        repair = (
            -row[
                "damped_minus_adamw"
            ]
        )

        print(
            f"Fold {int(row['fold']):02d} "
            f"baseline-gap="
            f"{baseline_gap:+.6f} "
            f"damping-repair="
            f"{repair:+.6f} "
            f"damped-vs-price="
            f"{row['damped_minus_price']:+.6f} "
            f"active="
            f"{row['damped_fraction']:.1%}"
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
        "Control     ->",
        CONTROL_FILE,
    )

    print(
        "Folds       ->",
        FOLD_FILE,
    )


if __name__ == "__main__":
    main()
