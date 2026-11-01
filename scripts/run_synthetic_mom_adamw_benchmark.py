from __future__ import annotations

import csv
import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_synthetic_latent_benchmark as synthetic
import run_synthetic_optimizer_stress_benchmark as stress

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.robust_gradient import (
    apply_coordinatewise_median_of_means,
)
from latent_information_dynamics.models.latent.synthetic_benchmark import (
    generate_synthetic_latent_dataset,
)


GROUPS = 5

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = stress.MAX_EPOCHS

BATCH_SIZE = stress.BATCH_SIZE


BASELINE_FILE = (
    stress.RUN_FILE
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/synthetic_mom_adamw"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "synthetic_mom_adamw_benchmark.json"
)

RUN_FILE = (
    OUTPUT_ROOT
    / "synthetic_mom_adamw_runs.csv"
)


def train_mom_adamw(
    data,
    *,
    seed: int,
) -> dict[str, object]:
    arrays = stress.validation_arrays(
        data
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
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 100000
    )

    best_validation_loss = (
        float(
            "inf"
        )
    )

    best_epoch = 1
    best_state = None

    robust_norms = []
    group_ratios = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        permutation = torch.randperm(
            arrays[
                "price_train"
            ].shape[
                0
            ],
            generator=generator,
        )

        for start in range(
            0,
            permutation.shape[
                0
            ],
            BATCH_SIZE,
        ):
            indices = permutation[
                start:
                start
                + BATCH_SIZE
            ]

            blocks = torch.tensor_split(
                indices,
                GROUPS,
            )

            if any(
                block.numel() == 0
                for block in blocks
            ):
                raise RuntimeError(
                    "Empty median-of-means block."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            losses = []

            for block in blocks:
                logits = model(
                    arrays[
                        "price_train"
                    ][
                        block
                    ],
                    arrays[
                        "flow_train"
                    ][
                        block
                    ],
                )

                losses.append(
                    criterion(
                        logits,
                        arrays[
                            "target_train"
                        ][
                            block
                        ],
                    )
                )

            snapshot = (
                apply_coordinatewise_median_of_means(
                    losses=losses,
                    parameters=
                        model.parameters(),
                )
            )

            robust_norms.append(
                snapshot
                .robust_gradient_norm
            )

            group_ratios.append(
                snapshot
                .max_to_median_norm_ratio
            )

            optimizer.step()

        validation_probability = (
            stress.predict_multimodal(
                model,
                arrays[
                    "price_validation"
                ],
                arrays[
                    "flow_validation"
                ],
            )
        )

        validation_loss = (
            synthetic.log_loss(
                arrays[
                    "target_validation"
                ],
                validation_probability,
            )
        )

        if (
            validation_loss
            < best_validation_loss
            - 1e-12
        ):
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = {
                key:
                    value
                    .detach()
                    .clone()
                for key, value
                in model
                .state_dict()
                .items()
            }

    if best_state is None:
        raise RuntimeError(
            "No selected model state."
        )

    model.load_state_dict(
        best_state
    )

    test_probability = (
        stress.predict_multimodal(
            model,
            arrays[
                "price_test"
            ],
            arrays[
                "flow_test"
            ],
        )
    )

    return {
        "best_epoch":
            best_epoch,
        "validation_log_loss":
            best_validation_loss,
        "test_probability":
            test_probability,
        "target_test":
            arrays[
                "target_test"
            ],
        "mean_robust_gradient_norm":
            float(
                np.mean(
                    robust_norms
                )
            ),
        "median_max_to_median_group_norm_ratio":
            float(
                np.median(
                    group_ratios
                )
            ),
        "p95_max_to_median_group_norm_ratio":
            float(
                np.quantile(
                    group_ratios,
                    0.95,
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

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Run the synthetic optimizer "
            "stress benchmark first."
        )

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    adamw = baseline[
        baseline[
            "optimizer"
        ]
        == "adamw"
    ].copy()

    if adamw.shape[0] != (
        len(
            stress.EVALUATION_SEEDS
        )
        * len(
            synthetic.SCENARIOS
        )
    ):
        raise RuntimeError(
            "Unexpected AdamW baseline rows."
        )

    baseline_lookup = {
        (
            int(
                row[
                    "seed"
                ]
            ),
            str(
                row[
                    "scenario"
                ]
            ),
        ):
            row
        for _, row
        in adamw.iterrows()
    }

    run_rows = []

    print(
        "MATCHED MoM-ADAMW BENCHMARK"
    )

    print(
        "==========================="
    )

    print(
        "Groups:",
        GROUPS,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print()

    for seed in (
        stress.EVALUATION_SEEDS
    ):
        for scenario_name, config in (
            synthetic.SCENARIOS.items()
        ):
            data = (
                generate_synthetic_latent_dataset(
                    config,
                    seed=seed,
                )
            )

            # Exactly the same initialization seed
            # used by the frozen optimizer benchmark.
            training_seed = (
                seed
                + 5000000
            )

            result = train_mom_adamw(
                data,
                seed=training_seed,
            )

            metrics = (
                synthetic.metric_set(
                    result[
                        "target_test"
                    ],
                    result[
                        "test_probability"
                    ],
                )
            )

            baseline_row = (
                baseline_lookup[
                    (
                        seed,
                        scenario_name,
                    )
                ]
            )

            adamw_ll = float(
                baseline_row[
                    "test_log_loss"
                ]
            )

            price_ll = float(
                baseline_row[
                    "price_log_loss"
                ]
            )

            row = {
                "seed":
                    seed,
                "scenario":
                    scenario_name,
                "optimizer":
                    "mom_adamw",
                "groups":
                    GROUPS,
                "learning_rate":
                    LEARNING_RATE,
                "selected_epoch":
                    result[
                        "best_epoch"
                    ],
                "validation_log_loss":
                    result[
                        "validation_log_loss"
                    ],
                "test_log_loss":
                    metrics[
                        "log_loss"
                    ],
                "test_brier":
                    metrics[
                        "brier"
                    ],
                "test_accuracy":
                    metrics[
                        "accuracy"
                    ],
                "adamw_log_loss":
                    adamw_ll,
                "price_log_loss":
                    price_ll,
                "mom_minus_adamw":
                    (
                        metrics[
                            "log_loss"
                        ]
                        - adamw_ll
                    ),
                "mom_minus_price":
                    (
                        metrics[
                            "log_loss"
                        ]
                        - price_ll
                    ),
                "mean_robust_gradient_norm":
                    result[
                        "mean_robust_gradient_norm"
                    ],
                "median_group_outlier_ratio":
                    result[
                        "median_max_to_median_group_norm_ratio"
                    ],
                "p95_group_outlier_ratio":
                    result[
                        "p95_max_to_median_group_norm_ratio"
                    ],
            }

            run_rows.append(
                row
            )

            print(
                f"seed={seed} "
                f"{scenario_name:30s} "
                f"AdamW="
                f"{adamw_ll:.5f} "
                f"MoM="
                f"{metrics['log_loss']:.5f} "
                f"delta="
                f"{row['mom_minus_adamw']:+.5f} "
                f"e="
                f"{int(result['best_epoch']):02d}"
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RUN_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                run_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            run_rows
        )

    summaries = {}

    print()
    print(
        "MoM-ADAMW ROBUST GRADIENT BENCHMARK"
    )

    print(
        "=================================="
    )

    for scenario_name in (
        synthetic.SCENARIOS
    ):
        rows = [
            row
            for row in run_rows
            if row[
                "scenario"
            ]
            == scenario_name
        ]

        mom = np.asarray(
            [
                row[
                    "test_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        adamw_values = np.asarray(
            [
                row[
                    "adamw_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        price = np.asarray(
            [
                row[
                    "price_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        versus_adamw = (
            mom
            - adamw_values
        )

        versus_price = (
            mom
            - price
        )

        adamw_summary = (
            stress.paired_summary(
                versus_adamw
            )
        )

        price_summary = (
            stress.paired_summary(
                versus_price
            )
        )

        summaries[
            scenario_name
        ] = {
            "adamw_mean_log_loss":
                float(
                    np.mean(
                        adamw_values
                    )
                ),
            "mom_adamw_mean_log_loss":
                float(
                    np.mean(
                        mom
                    )
                ),
            "mom_vs_adamw":
                adamw_summary,
            "mom_vs_price":
                price_summary,
            "mean_selected_epoch":
                float(
                    np.mean(
                        [
                            row[
                                "selected_epoch"
                            ]
                            for row in rows
                        ]
                    )
                ),
            "median_group_outlier_ratio":
                float(
                    np.median(
                        [
                            row[
                                "median_group_outlier_ratio"
                            ]
                            for row in rows
                        ]
                    )
                ),
            "mean_p95_group_outlier_ratio":
                float(
                    np.mean(
                        [
                            row[
                                "p95_group_outlier_ratio"
                            ]
                            for row in rows
                        ]
                    )
                ),
        }

        print()
        print(
            scenario_name
        )

        print(
            "-" * len(
                scenario_name
            )
        )

        print(
            "AdamW LL:",
            f"{np.mean(adamw_values):.6f}",
        )

        print(
            "MoM-AdamW LL:",
            f"{np.mean(mom):.6f}",
        )

        print(
            "MoM - AdamW:",
            f"{adamw_summary['mean']:+.6f}",
        )

        print(
            "SE:",
            f"{adamw_summary['standard_error']:.6f}",
        )

        print(
            "t:",
            f"{adamw_summary['t_statistic']:+.3f}",
        )

        print(
            "normal p:",
            f"{adamw_summary['normal_p_value']:.6f}",
        )

        print(
            "MoM wins AdamW:",
            f"{adamw_summary['wins']}/"
            f"{adamw_summary['observations']}",
        )

        print(
            "MoM - Price:",
            f"{price_summary['mean']:+.6f}",
        )

        print(
            "Median group outlier ratio:",
            f"{summaries[scenario_name]['median_group_outlier_ratio']:.3f}",
        )

        print(
            "Mean p95 group outlier ratio:",
            f"{summaries[scenario_name]['mean_p95_group_outlier_ratio']:.3f}",
        )

    print()
    print(
        "STRESS PENALTIES"
    )

    print(
        "----------------"
    )

    penalties = {}

    for optimizer_name in (
        "adamw",
        "mom_adamw",
    ):
        stationary = {}

        for row in run_rows:
            if row[
                "scenario"
            ] != "stationary_gaussian":
                continue

            value = (
                row[
                    "adamw_log_loss"
                ]
                if optimizer_name
                == "adamw"
                else row[
                    "test_log_loss"
                ]
            )

            stationary[
                int(
                    row[
                        "seed"
                    ]
                )
            ] = float(
                value
            )

        penalties[
            optimizer_name
        ] = {}

        print()
        print(
            optimizer_name
        )

        for scenario_name in (
            "stationary_heavy_tail",
            "flow_scale_shift",
            "flow_sign_shift_gaussian",
            "flow_sign_shift_heavy_tail",
        ):
            scenario_rows = [
                row
                for row in run_rows
                if row[
                    "scenario"
                ]
                == scenario_name
            ]

            values = []

            for row in scenario_rows:
                value = (
                    row[
                        "adamw_log_loss"
                    ]
                    if optimizer_name
                    == "adamw"
                    else row[
                        "test_log_loss"
                    ]
                )

                values.append(
                    float(
                        value
                    )
                    - stationary[
                        int(
                            row[
                                "seed"
                            ]
                        )
                    ]
                )

            summary = (
                stress.paired_summary(
                    np.asarray(
                        values,
                        dtype=np.float64,
                    )
                )
            )

            penalties[
                optimizer_name
            ][
                scenario_name
            ] = summary

            print(
                f"{scenario_name:30s} "
                f"penalty="
                f"{summary['mean']:+.6f}"
            )

    results = {
        "experiment":
            "synthetic-mom-adamw-benchmark",
        "status":
            (
                "Developmental controlled robust-gradient "
                "benchmark. No stress-scenario tuning."
            ),
        "protocol": {
            "gradient_estimator":
                (
                    "5-block coordinatewise "
                    "median-of-means"
                ),
            "optimizer":
                "AdamW",
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "optimizer_hyperparameters":
                (
                    "Identical to previously frozen "
                    "AdamW configuration."
                ),
            "initialization_seed":
                (
                    "Matched to standard AdamW "
                    "evaluation."
                ),
            "minibatch_order":
                (
                    "Matched to standard AdamW "
                    "evaluation."
                ),
            "stress_scenario_tuning":
                False,
        },
        "summaries":
            summaries,
        "stress_penalties":
            penalties,
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
        "Results ->",
        RESULTS_FILE,
    )

    print(
        "Runs    ->",
        RUN_FILE,
    )


if __name__ == "__main__":
    main()
