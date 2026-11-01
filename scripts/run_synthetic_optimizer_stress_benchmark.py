from __future__ import annotations

import copy
import csv
import json
import math

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_synthetic_latent_benchmark as synthetic

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    PriceLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.synthetic_benchmark import (
    generate_synthetic_latent_dataset,
)


CALIBRATION_SEEDS = (
    910001,
    910002,
    910003,
    910004,
)

EVALUATION_SEEDS = tuple(
    range(
        920001,
        920013,
    )
)

MAX_EPOCHS = 50
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
EPSILON = 1e-12


OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/synthetic_optimizer_stress"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "synthetic_optimizer_stress_benchmark.json"
)

CALIBRATION_FILE = (
    OUTPUT_ROOT
    / "synthetic_optimizer_calibration.csv"
)

RUN_FILE = (
    OUTPUT_ROOT
    / "synthetic_optimizer_stress_runs.csv"
)


@dataclass(frozen=True)
class OptimizerCandidate:
    family: str
    learning_rate: float
    clip_norm: float | None = None

    @property
    def key(self) -> str:
        if self.clip_norm is None:
            return (
                f"{self.family}"
                f"_lr{self.learning_rate:g}"
            )

        return (
            f"{self.family}"
            f"_lr{self.learning_rate:g}"
            f"_clip{self.clip_norm:g}"
        )


CANDIDATES = {
    "adamw": [
        OptimizerCandidate(
            "adamw",
            3e-4,
        ),
        OptimizerCandidate(
            "adamw",
            1e-3,
        ),
        OptimizerCandidate(
            "adamw",
            3e-3,
        ),
    ],

    "sgd_momentum": [
        OptimizerCandidate(
            "sgd_momentum",
            3e-3,
        ),
        OptimizerCandidate(
            "sgd_momentum",
            1e-2,
        ),
        OptimizerCandidate(
            "sgd_momentum",
            3e-2,
        ),
        OptimizerCandidate(
            "sgd_momentum",
            1e-1,
        ),
    ],

    "normalized_sgd": [
        OptimizerCandidate(
            "normalized_sgd",
            1e-3,
        ),
        OptimizerCandidate(
            "normalized_sgd",
            3e-3,
        ),
        OptimizerCandidate(
            "normalized_sgd",
            1e-2,
        ),
        OptimizerCandidate(
            "normalized_sgd",
            3e-2,
        ),
    ],

    "clipped_sgd": [
        OptimizerCandidate(
            "clipped_sgd",
            learning_rate,
            clip_norm,
        )
        for learning_rate in (
            3e-3,
            1e-2,
            3e-2,
        )
        for clip_norm in (
            0.10,
            0.25,
            0.50,
        )
    ],
}


def make_optimizer(
    model: nn.Module,
    candidate: OptimizerCandidate,
):
    if candidate.family == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=candidate.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

    if candidate.family == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=candidate.learning_rate,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    if candidate.family in {
        "normalized_sgd",
        "clipped_sgd",
    }:
        return torch.optim.SGD(
            model.parameters(),
            lr=candidate.learning_rate,
            momentum=0.0,
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(
        f"Unknown optimizer family: "
        f"{candidate.family}"
    )


def global_gradient_norm(
    model: nn.Module,
) -> float:
    squared = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient = (
            parameter.grad
            .detach()
            .to(
                dtype=torch.float64,
            )
        )

        squared += float(
            torch.sum(
                gradient * gradient
            ).item()
        )

    return math.sqrt(
        squared
    )


def transform_gradients(
    model: nn.Module,
    candidate: OptimizerCandidate,
) -> tuple[
    float,
    float,
]:
    raw_norm = global_gradient_norm(
        model
    )

    scale = 1.0

    if candidate.family == "normalized_sgd":
        if raw_norm > EPSILON:
            scale = (
                1.0
                / raw_norm
            )

    elif candidate.family == "clipped_sgd":
        if candidate.clip_norm is None:
            raise ValueError(
                "clipped_sgd requires clip_norm."
            )

        if raw_norm > candidate.clip_norm:
            scale = (
                candidate.clip_norm
                / raw_norm
            )

    if scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            parameter.grad.mul_(
                scale
            )

    return (
        raw_norm,
        scale,
    )


def tensor(
    values: np.ndarray,
) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def validation_arrays(
    data,
):
    (
        price_train,
        price_validation,
        price_test,
        flow_train,
        flow_validation,
        flow_test,
        target_train,
        target_validation,
        target_test,
    ) = synthetic.prepare_arrays(
        data,
        include_flow=True,
    )

    return {
        "price_train":
            tensor(
                price_train
            ),
        "price_validation":
            tensor(
                price_validation
            ),
        "price_test":
            tensor(
                price_test
            ),
        "flow_train":
            tensor(
                flow_train
            ),
        "flow_validation":
            tensor(
                flow_validation
            ),
        "flow_test":
            tensor(
                flow_test
            ),
        "target_train":
            tensor(
                target_train
            ),
        "target_validation":
            np.asarray(
                target_validation,
                dtype=np.float64,
            ),
        "target_test":
            np.asarray(
                target_test,
                dtype=np.float64,
            ),
    }


def predict_multimodal(
    model: MultimodalLatentModel,
    price_x: torch.Tensor,
    flow_x: torch.Tensor,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        probability = torch.sigmoid(
            model(
                price_x,
                flow_x,
            )
        )

    return np.asarray(
        probability.cpu().numpy(),
        dtype=np.float64,
    )


def train_multimodal_candidate(
    data,
    *,
    candidate: OptimizerCandidate,
    seed: int,
    return_test: bool,
) -> dict[str, object]:
    arrays = validation_arrays(
        data
    )

    set_research_seed(
        seed
    )

    model = (
        MultimodalLatentModel()
    )

    optimizer = make_optimizer(
        model,
        candidate,
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

    raw_norms = []
    scales = []

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
            permutation.shape[0],
            BATCH_SIZE,
        ):
            indices = permutation[
                start:
                start + BATCH_SIZE
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                arrays[
                    "price_train"
                ][
                    indices
                ],
                arrays[
                    "flow_train"
                ][
                    indices
                ],
            )

            loss = criterion(
                logits,
                arrays[
                    "target_train"
                ][
                    indices
                ],
            )

            loss.backward()

            (
                raw_norm,
                scale,
            ) = transform_gradients(
                model,
                candidate,
            )

            raw_norms.append(
                raw_norm
            )

            scales.append(
                scale
            )

            optimizer.step()

        validation_probability = (
            predict_multimodal(
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
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No best model state selected."
        )

    result = {
        "best_epoch":
            best_epoch,
        "validation_log_loss":
            best_validation_loss,
        "mean_raw_gradient_norm":
            float(
                np.mean(
                    raw_norms
                )
            ),
        "p99_raw_gradient_norm":
            float(
                np.quantile(
                    raw_norms,
                    0.99,
                )
            ),
        "mean_gradient_scale":
            float(
                np.mean(
                    scales
                )
            ),
        "transformed_fraction":
            float(
                np.mean(
                    np.asarray(
                        scales
                    )
                    < 1.0
                )
            ),
    }

    if return_test:
        model.load_state_dict(
            best_state
        )

        test_probability = (
            predict_multimodal(
                model,
                arrays[
                    "price_test"
                ],
                arrays[
                    "flow_test"
                ],
            )
        )

        result[
            "test_probability"
        ] = test_probability

        result[
            "target_test"
        ] = arrays[
            "target_test"
        ]

    return result


def train_price_reference(
    data,
    *,
    seed: int,
) -> dict[str, object]:
    (
        price_train,
        price_validation,
        price_test,
        _,
        _,
        _,
        target_train,
        target_validation,
        target_test,
    ) = synthetic.prepare_arrays(
        data,
        include_flow=False,
    )

    price_train = tensor(
        price_train
    )

    price_validation = tensor(
        price_validation
    )

    price_test = tensor(
        price_test
    )

    target_train_tensor = tensor(
        target_train
    )

    set_research_seed(
        seed
    )

    model = PriceLatentModel()

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
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
        seed + 200000
    )

    best_loss = float(
        "inf"
    )

    best_epoch = 1
    best_state = None

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        permutation = torch.randperm(
            price_train.shape[0],
            generator=generator,
        )

        for start in range(
            0,
            permutation.shape[0],
            BATCH_SIZE,
        ):
            indices = permutation[
                start:
                start + BATCH_SIZE
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                price_train[
                    indices
                ]
            )

            loss = criterion(
                logits,
                target_train_tensor[
                    indices
                ],
            )

            loss.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_probability = (
                torch.sigmoid(
                    model(
                        price_validation
                    )
                )
                .cpu()
                .numpy()
            )

        validation_loss = (
            synthetic.log_loss(
                target_validation,
                validation_probability,
            )
        )

        if (
            validation_loss
            < best_loss
            - 1e-12
        ):
            best_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = {
                key:
                    value
                    .detach()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Price model selection failed."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():
        probability = (
            torch.sigmoid(
                model(
                    price_test
                )
            )
            .cpu()
            .numpy()
        )

    return {
        "best_epoch":
            best_epoch,
        "probability":
            np.asarray(
                probability,
                dtype=np.float64,
            ),
        "target":
            np.asarray(
                target_test,
                dtype=np.float64,
            ),
    }


def calibrate() -> tuple[
    dict[str, OptimizerCandidate],
    list[dict[str, object]],
]:
    stationary = (
        synthetic.SCENARIOS[
            "stationary_gaussian"
        ]
    )

    rows = []

    print(
        "OPTIMIZER CALIBRATION"
    )

    print(
        "====================="
    )

    for family, candidates in (
        CANDIDATES.items()
    ):
        for candidate in candidates:
            losses = []

            epochs = []

            for seed in (
                CALIBRATION_SEEDS
            ):
                data = (
                    generate_synthetic_latent_dataset(
                        stationary,
                        seed=seed,
                    )
                )

                result = (
                    train_multimodal_candidate(
                        data,
                        candidate=candidate,
                        seed=
                            seed
                            + 3000000,
                        return_test=False,
                    )
                )

                losses.append(
                    float(
                        result[
                            "validation_log_loss"
                        ]
                    )
                )

                epochs.append(
                    int(
                        result[
                            "best_epoch"
                        ]
                    )
                )

            row = {
                "family":
                    family,
                "candidate":
                    candidate.key,
                "learning_rate":
                    candidate.learning_rate,
                "clip_norm":
                    candidate.clip_norm,
                "mean_validation_log_loss":
                    float(
                        np.mean(
                            losses
                        )
                    ),
                "mean_selected_epoch":
                    float(
                        np.mean(
                            epochs
                        )
                    ),
            }

            rows.append(
                row
            )

            print(
                f"{candidate.key:36s} "
                f"ValLL="
                f"{row['mean_validation_log_loss']:.6f} "
                f"Epoch="
                f"{row['mean_selected_epoch']:.1f}"
            )

    selected = {}

    for family, candidates in (
        CANDIDATES.items()
    ):
        family_rows = [
            row
            for row in rows
            if row[
                "family"
            ]
            == family
        ]

        best = min(
            family_rows,
            key=lambda row:
                (
                    row[
                        "mean_validation_log_loss"
                    ],
                    row[
                        "learning_rate"
                    ],
                    float(
                        "inf"
                    )
                    if row[
                        "clip_norm"
                    ] is None
                    else row[
                        "clip_norm"
                    ],
                ),
        )

        selected[
            family
        ] = next(
            candidate
            for candidate in candidates
            if candidate.key
            == best[
                "candidate"
            ]
        )

    print()
    print(
        "FROZEN OPTIMIZER CONFIGURATIONS"
    )

    print(
        "-------------------------------"
    )

    for family, candidate in (
        selected.items()
    ):
        print(
            family,
            "->",
            candidate.key,
        )

    return (
        selected,
        rows,
    )


def paired_summary(
    values: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean_value = float(
        np.mean(
            values
        )
    )

    standard_error = float(
        np.std(
            values,
            ddof=1,
        )
        / np.sqrt(
            values.size
        )
    )

    if standard_error > 0.0:
        statistic = (
            mean_value
            / standard_error
        )

        p_value = math.erfc(
            abs(
                statistic
            )
            / math.sqrt(
                2.0
            )
        )
    else:
        statistic = 0.0
        p_value = 1.0

    return {
        "mean":
            mean_value,
        "standard_error":
            standard_error,
        "t_statistic":
            statistic,
        "normal_p_value":
            p_value,
        "wins":
            int(
                np.sum(
                    values < 0.0
                )
            ),
        "observations":
            int(
                values.size
            ),
    }


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    (
        selected,
        calibration_rows,
    ) = calibrate()

    run_rows = []

    print()
    print(
        "FROZEN OPTIMIZER STRESS EVALUATION"
    )

    print(
        "=================================="
    )

    for seed in EVALUATION_SEEDS:
        stationary_data = (
            generate_synthetic_latent_dataset(
                synthetic.SCENARIOS[
                    "stationary_gaussian"
                ],
                seed=seed,
            )
        )

        price_result = (
            train_price_reference(
                stationary_data,
                seed=
                    seed
                    + 4000000,
            )
        )

        price_metrics = (
            synthetic.metric_set(
                price_result[
                    "target"
                ],
                price_result[
                    "probability"
                ],
            )
        )

        for scenario_name, config in (
            synthetic.SCENARIOS.items()
        ):
            data = (
                generate_synthetic_latent_dataset(
                    config,
                    seed=seed,
                )
            )

            for family_index, (
                family,
                candidate,
            ) in enumerate(
                selected.items()
            ):
                # Same initialization and minibatch
                # sequence across optimizer families.
                training_seed = (
                    seed
                    + 5000000
                )

                result = (
                    train_multimodal_candidate(
                        data,
                        candidate=candidate,
                        seed=training_seed,
                        return_test=True,
                    )
                )

                test_metrics = (
                    synthetic.metric_set(
                        result[
                            "target_test"
                        ],
                        result[
                            "test_probability"
                        ],
                    )
                )

                row = {
                    "seed":
                        seed,
                    "scenario":
                        scenario_name,
                    "optimizer":
                        family,
                    "candidate":
                        candidate.key,
                    "selected_epoch":
                        result[
                            "best_epoch"
                        ],
                    "validation_log_loss":
                        result[
                            "validation_log_loss"
                        ],
                    "test_log_loss":
                        test_metrics[
                            "log_loss"
                        ],
                    "test_brier":
                        test_metrics[
                            "brier"
                        ],
                    "test_accuracy":
                        test_metrics[
                            "accuracy"
                        ],
                    "price_log_loss":
                        price_metrics[
                            "log_loss"
                        ],
                    "optimizer_minus_price":
                        (
                            test_metrics[
                                "log_loss"
                            ]
                            - price_metrics[
                                "log_loss"
                            ]
                        ),
                    "mean_raw_gradient_norm":
                        result[
                            "mean_raw_gradient_norm"
                        ],
                    "p99_raw_gradient_norm":
                        result[
                            "p99_raw_gradient_norm"
                        ],
                    "mean_gradient_scale":
                        result[
                            "mean_gradient_scale"
                        ],
                    "transformed_fraction":
                        result[
                            "transformed_fraction"
                        ],
                }

                run_rows.append(
                    row
                )

                print(
                    f"seed={seed} "
                    f"{scenario_name:30s} "
                    f"{family:15s} "
                    f"LL="
                    f"{row['test_log_loss']:.5f} "
                    f"vsPrice="
                    f"{row['optimizer_minus_price']:+.5f} "
                    f"e="
                    f"{int(row['selected_epoch']):02d}"
                )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CALIBRATION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                calibration_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            calibration_rows
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

    print()
    print(
        "SYNTHETIC OPTIMIZER BENCHMARK"
    )

    print(
        "============================="
    )

    summaries = {}

    optimizer_names = list(
        selected.keys()
    )

    for scenario_name in (
        synthetic.SCENARIOS
    ):
        print()
        print(
            scenario_name
        )

        print(
            "-" * len(
                scenario_name
            )
        )

        scenario_rows = [
            row
            for row in run_rows
            if row[
                "scenario"
            ]
            == scenario_name
        ]

        adamw_by_seed = {
            int(
                row[
                    "seed"
                ]
            ):
                float(
                    row[
                        "test_log_loss"
                    ]
                )
            for row in scenario_rows
            if row[
                "optimizer"
            ]
            == "adamw"
        }

        scenario_summary = {}

        price_values = np.asarray(
            [
                row[
                    "price_log_loss"
                ]
                for row in scenario_rows
                if row[
                    "optimizer"
                ]
                == "adamw"
            ],
            dtype=np.float64,
        )

        print(
            "Price reference:",
            f"{np.mean(price_values):.6f}"
        )

        for optimizer_name in (
            optimizer_names
        ):
            rows = [
                row
                for row in scenario_rows
                if row[
                    "optimizer"
                ]
                == optimizer_name
            ]

            values = np.asarray(
                [
                    row[
                        "test_log_loss"
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

            vs_price = np.asarray(
                [
                    row[
                        "optimizer_minus_price"
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

            vs_adamw = np.asarray(
                [
                    row[
                        "test_log_loss"
                    ]
                    - adamw_by_seed[
                        int(
                            row[
                                "seed"
                            ]
                        )
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

            price_summary = (
                paired_summary(
                    vs_price
                )
            )

            adamw_summary = (
                paired_summary(
                    vs_adamw
                )
            )

            scenario_summary[
                optimizer_name
            ] = {
                "mean_log_loss":
                    float(
                        np.mean(
                            values
                        )
                    ),
                "vs_price":
                    price_summary,
                "vs_adamw":
                    adamw_summary,
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
                "mean_gradient_scale":
                    float(
                        np.mean(
                            [
                                row[
                                    "mean_gradient_scale"
                                ]
                                for row in rows
                            ]
                        )
                    ),
                "mean_transformed_fraction":
                    float(
                        np.mean(
                            [
                                row[
                                    "transformed_fraction"
                                ]
                                for row in rows
                            ]
                        )
                    ),
            }

            print(
                f"{optimizer_name:15s} "
                f"LL="
                f"{np.mean(values):.6f} "
                f"vsPrice="
                f"{price_summary['mean']:+.6f} "
                f"vsAdamW="
                f"{adamw_summary['mean']:+.6f} "
                f"winsAdamW="
                f"{adamw_summary['wins']}/"
                f"{adamw_summary['observations']}"
            )

        summaries[
            scenario_name
        ] = (
            scenario_summary
        )

    print()
    print(
        "STRESS PENALTIES RELATIVE TO STATIONARY GAUSSIAN"
    )

    print(
        "-----------------------------------------------"
    )

    stress_penalties = {}

    for optimizer_name in (
        optimizer_names
    ):
        optimizer_rows = [
            row
            for row in run_rows
            if row[
                "optimizer"
            ]
            == optimizer_name
        ]

        stationary_by_seed = {
            int(
                row[
                    "seed"
                ]
            ):
                float(
                    row[
                        "test_log_loss"
                    ]
                )
            for row in optimizer_rows
            if row[
                "scenario"
            ]
            == "stationary_gaussian"
        }

        stress_penalties[
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
            stressed = [
                row
                for row in optimizer_rows
                if row[
                    "scenario"
                ]
                == scenario_name
            ]

            penalty = np.asarray(
                [
                    float(
                        row[
                            "test_log_loss"
                        ]
                    )
                    - stationary_by_seed[
                        int(
                            row[
                                "seed"
                            ]
                        )
                    ]
                    for row in stressed
                ],
                dtype=np.float64,
            )

            summary = (
                paired_summary(
                    penalty
                )
            )

            stress_penalties[
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
            "synthetic-optimizer-stress-benchmark",
        "status":
            (
                "Developmental controlled optimizer "
                "benchmark with optimizer hyperparameters "
                "calibrated only on independent stationary "
                "Gaussian seeds and frozen before stress "
                "evaluation."
            ),
        "calibration_seeds":
            list(
                CALIBRATION_SEEDS
            ),
        "evaluation_seeds":
            list(
                EVALUATION_SEEDS
            ),
        "frozen_optimizer_configuration": {
            family: {
                "candidate":
                    candidate.key,
                "learning_rate":
                    candidate.learning_rate,
                "clip_norm":
                    candidate.clip_norm,
            }
            for family, candidate
            in selected.items()
        },
        "summaries":
            summaries,
        "stress_penalties":
            stress_penalties,
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
        "Results     ->",
        RESULTS_FILE,
    )

    print(
        "Calibration ->",
        CALIBRATION_FILE,
    )

    print(
        "Runs        ->",
        RUN_FILE,
    )


if __name__ == "__main__":
    main()
