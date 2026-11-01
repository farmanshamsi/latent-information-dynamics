from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_nested_latent_neural_walk_forward as base
import run_nested_clipped_adamw_walk_forward as clipping

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.gradient_interference import (
    capture_shared_gradient_interference,
)
from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
)


NESTED_RESULTS_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward/"
    "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

INTERVENTION_FOLD_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/modality_intervention/"
    "multimodal_modality_intervention_folds.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/gradient_interference"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "multimodal_gradient_interference.json"
)

STEP_FILE = (
    OUTPUT_ROOT
    / "multimodal_gradient_interference_steps.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "multimodal_gradient_interference_folds.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "multimodal_gradient_interference_correlations.csv"
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


def finite(
    values: np.ndarray,
) -> np.ndarray:
    return values[
        np.isfinite(
            values
        )
    ]


def summarize_fold(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    cosine = finite(
        frame[
            "price_flow_cosine"
        ].to_numpy(
            dtype=np.float64
        )
    )

    price_norm = finite(
        frame[
            "price_context_shared_norm"
        ].to_numpy(
            dtype=np.float64
        )
    )

    flow_norm = finite(
        frame[
            "flow_context_shared_norm"
        ].to_numpy(
            dtype=np.float64
        )
    )

    ratio = (
        flow_norm
        / np.maximum(
            price_norm,
            1e-12,
        )
    )

    progress = frame[
        "training_progress"
    ].to_numpy(
        dtype=np.float64
    )

    early = finite(
        frame.loc[
            progress <= 0.25,
            "price_flow_cosine",
        ].to_numpy(
            dtype=np.float64
        )
    )

    late = finite(
        frame.loc[
            progress >= 0.75,
            "price_flow_cosine",
        ].to_numpy(
            dtype=np.float64
        )
    )

    return {
        "steps":
            int(
                frame.shape[0]
            ),
        "mean_price_flow_cosine":
            float(
                np.mean(cosine)
            ),
        "median_price_flow_cosine":
            float(
                np.median(cosine)
            ),
        "p10_price_flow_cosine":
            float(
                np.quantile(
                    cosine,
                    0.10,
                )
            ),
        "p05_price_flow_cosine":
            float(
                np.quantile(
                    cosine,
                    0.05,
                )
            ),
        "negative_cosine_fraction":
            float(
                np.mean(
                    cosine < 0.0
                )
            ),
        "strong_conflict_fraction":
            float(
                np.mean(
                    cosine < -0.25
                )
            ),
        "mean_price_context_norm":
            float(
                np.mean(
                    price_norm
                )
            ),
        "mean_flow_context_norm":
            float(
                np.mean(
                    flow_norm
                )
            ),
        "mean_flow_price_norm_ratio":
            float(
                np.mean(
                    ratio
                )
            ),
        "median_flow_price_norm_ratio":
            float(
                np.median(
                    ratio
                )
            ),
        "early_mean_cosine":
            float(
                np.mean(
                    early
                )
            ),
        "late_mean_cosine":
            float(
                np.mean(
                    late
                )
            ),
        "late_minus_early_cosine":
            float(
                np.mean(
                    late
                )
                - np.mean(
                    early
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

    intervention = pd.read_csv(
        INTERVENTION_FOLD_FILE
    )

    intervention_by_fold = {
        int(row["fold"]):
            row
        for _, row
        in intervention.iterrows()
    }

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

    step_rows = []
    reproduction_errors = []

    print(
        "MULTIMODAL SHARED-GRADIENT INTERFERENCE"
    )

    print(
        "======================================"
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

        epochs = selected_epochs[
            fold
        ]

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

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            seed + 400000
        )

        global_step = 0

        batches_per_epoch = int(
            np.ceil(
                price_train.shape[0]
                / base.BATCH_SIZE
            )
        )

        total_steps = (
            batches_per_epoch
            * epochs
        )

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

                optimizer.zero_grad(
                    set_to_none=True
                )

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

                logits = model(
                    batch_price,
                    batch_flow,
                )

                loss = criterion(
                    logits,
                    batch_target,
                )

                loss.backward()

                diagnostic = (
                    capture_shared_gradient_interference(
                        model,
                        batch_price,
                        batch_flow,
                        batch_target,
                    )
                )

                global_step += 1

                row = {
                    "fold":
                        fold,
                    "epoch":
                        epoch,
                    "batch":
                        batch,
                    "global_step":
                        global_step,
                    "training_progress":
                        global_step
                        / total_steps,
                    "batch_observations":
                        int(
                            indices.shape[0]
                        ),
                    "full_loss":
                        float(
                            loss
                            .detach()
                            .item()
                        ),
                }

                row.update(
                    diagnostic.as_dict()
                )

                step_rows.append(
                    row
                )

                optimizer.step()

        model.eval()

        with torch.no_grad():
            reproduced = (
                model.predict_proba(
                    price_test,
                    flow_test,
                )
                .cpu()
                .numpy()
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
                    reproduced
                    - stored
                )
            )
        )

        reproduction_errors.append(
            reproduction_error
        )

        if reproduction_error > 1e-10:
            raise RuntimeError(
                "Training-path reproduction failed "
                f"in fold {fold}: "
                f"{reproduction_error}"
            )

        fold_frame = pd.DataFrame(
            [
                row
                for row in step_rows
                if int(
                    row["fold"]
                ) == fold
            ]
        )

        summary = summarize_fold(
            fold_frame
        )

        print(
            f"Fold {fold:02d}: "
            f"e={epochs:2d} "
            f"mean-cos="
            f"{summary['mean_price_flow_cosine']:+.3f} "
            f"neg="
            f"{summary['negative_cosine_fraction']:.1%} "
            f"strong="
            f"{summary['strong_conflict_fraction']:.1%} "
            f"late-early="
            f"{summary['late_minus_early_cosine']:+.3f}"
        )

    steps = pd.DataFrame(
        step_rows
    )

    fold_rows = []

    for fold in range(20):
        fold_frame = steps[
            steps[
                "fold"
            ]
            == fold
        ]

        summary = summarize_fold(
            fold_frame
        )

        intervention_row = (
            intervention_by_fold[
                fold
            ]
        )

        summary.update(
            {
                "fold":
                    fold,
                "epochs":
                    selected_epochs[
                        fold
                    ],
                "multimodal_minus_price_nn":
                    float(
                        intervention_row[
                            "full_minus_price_nn"
                        ]
                    ),
                "flow_harm":
                    -float(
                        intervention_row[
                            "flow_neutral_minus_full"
                        ]
                    ),
            }
        )

        fold_rows.append(
            summary
        )

    folds = pd.DataFrame(
        fold_rows
    )

    diagnostic_columns = [
        "mean_price_flow_cosine",
        "median_price_flow_cosine",
        "p10_price_flow_cosine",
        "p05_price_flow_cosine",
        "negative_cosine_fraction",
        "strong_conflict_fraction",
        "mean_price_context_norm",
        "mean_flow_context_norm",
        "mean_flow_price_norm_ratio",
        "median_flow_price_norm_ratio",
        "early_mean_cosine",
        "late_mean_cosine",
        "late_minus_early_cosine",
    ]

    correlation_rows = []

    for target_name in (
        "multimodal_minus_price_nn",
        "flow_harm",
    ):
        target = folds[
            target_name
        ].to_numpy(
            dtype=np.float64
        )

        for diagnostic in diagnostic_columns:
            values = folds[
                diagnostic
            ].to_numpy(
                dtype=np.float64
            )

            correlation_rows.append(
                {
                    "target":
                        target_name,
                    "diagnostic":
                        diagnostic,
                    "pearson":
                        correlation(
                            values,
                            target,
                            method="pearson",
                        ),
                    "spearman":
                        correlation(
                            values,
                            target,
                            method="spearman",
                        ),
                }
            )

    correlations = pd.DataFrame(
        correlation_rows
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps.to_csv(
        STEP_FILE,
        index=False,
    )

    folds.to_csv(
        FOLD_FILE,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    results = {
        "experiment":
            "multimodal-counterfactual-shared-gradient-interference",
        "status":
            (
                "Developmental mechanistic diagnostic. "
                "Counterfactual modality-conditioned "
                "gradients are not interpreted as an "
                "exact additive decomposition of the "
                "full nonlinear model gradient."
            ),
        "protocol": {
            "shared_parameters":
                "fusion + prediction head",
            "price_context":
                "(standardized Price, zero Flow)",
            "flow_context":
                "(zero Price, standardized Flow)",
            "training_path_reproduced":
                True,
            "maximum_probability_reproduction_error":
                max(
                    reproduction_errors
                ),
            "point_in_time_training":
                True,
        },
        "steps":
            int(
                steps.shape[0]
            ),
        "folds":
            20,
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
        "AGGREGATE INTERFERENCE"
    )

    print(
        "----------------------"
    )

    finite_cosine = finite(
        steps[
            "price_flow_cosine"
        ].to_numpy(
            dtype=np.float64
        )
    )

    print(
        "Steps:",
        finite_cosine.shape[0],
    )

    print(
        "Mean Price/Flow cosine:",
        f"{np.mean(finite_cosine):+.4f}",
    )

    print(
        "Median Price/Flow cosine:",
        f"{np.median(finite_cosine):+.4f}",
    )

    print(
        "Negative cosine fraction:",
        f"{np.mean(finite_cosine < 0.0):.2%}",
    )

    print(
        "Strong conflict (< -0.25):",
        f"{np.mean(finite_cosine < -0.25):.2%}",
    )

    for target_name, title in (
        (
            "multimodal_minus_price_nn",
            "INTERFERENCE VS MULTIMODAL FAILURE",
        ),
        (
            "flow_harm",
            "INTERFERENCE VS FLOW HARM",
        ),
    ):
        print()
        print(
            title
        )

        print(
            "-" * len(title)
        )

        subset = correlations[
            correlations[
                "target"
            ]
            == target_name
        ].copy()

        subset[
            "absolute_spearman"
        ] = (
            subset[
                "spearman"
            ].abs()
        )

        subset = subset.sort_values(
            "absolute_spearman",
            ascending=False,
        )

        for _, row in (
            subset.head(
                10
            ).iterrows()
        ):
            print(
                f"{row['diagnostic']:32s} "
                f"Pearson="
                f"{row['pearson']:+.3f} "
                f"Spearman="
                f"{row['spearman']:+.3f}"
            )

    print()
    print(
        "WORST MULTIMODAL FOLDS"
    )

    print(
        "----------------------"
    )

    worst = folds.sort_values(
        "multimodal_minus_price_nn",
        ascending=False,
    ).head(
        8
    )

    for _, row in (
        worst.iterrows()
    ):
        print(
            f"Fold {int(row['fold']):02d} "
            f"gap="
            f"{row['multimodal_minus_price_nn']:+.6f} "
            f"FlowHarm="
            f"{row['flow_harm']:+.6f} "
            f"meanCos="
            f"{row['mean_price_flow_cosine']:+.3f} "
            f"neg="
            f"{row['negative_cosine_fraction']:.1%} "
            f"strong="
            f"{row['strong_conflict_fraction']:.1%} "
            f"lateEarly="
            f"{row['late_minus_early_cosine']:+.3f}"
        )

    print()
    print(
        "Results      ->",
        RESULTS_FILE,
    )

    print(
        "Step stats   ->",
        STEP_FILE,
    )

    print(
        "Fold stats   ->",
        FOLD_FILE,
    )

    print(
        "Correlations ->",
        CORRELATION_FILE,
    )


if __name__ == "__main__":
    main()
