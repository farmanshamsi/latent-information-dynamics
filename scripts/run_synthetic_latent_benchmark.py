from __future__ import annotations

import csv
import json
import math

from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    PriceLatentModel,
    set_research_seed,
)
from latent_information_dynamics.models.latent.synthetic_benchmark import (
    SyntheticLatentConfig,
    generate_synthetic_latent_dataset,
)


BASE_SEED = 20261027

REPETITIONS = 12

MAX_EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


SCENARIOS = {
    "stationary_gaussian":
        SyntheticLatentConfig(),

    "stationary_heavy_tail":
        SyntheticLatentConfig(
            flow_heavy_tail_df=1.5,
        ),

    "flow_sign_shift_gaussian":
        SyntheticLatentConfig(
            flow_test_loading_multiplier=
                -1.0,
        ),

    "flow_sign_shift_heavy_tail":
        SyntheticLatentConfig(
            flow_test_loading_multiplier=
                -1.0,
            flow_heavy_tail_df=1.5,
        ),

    "flow_scale_shift":
        SyntheticLatentConfig(
            flow_test_noise_multiplier=
                4.0,
        ),
}


OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/synthetic_benchmark"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "synthetic_latent_benchmark.json"
)

RUN_FILE = (
    OUTPUT_ROOT
    / "synthetic_latent_benchmark_runs.csv"
)


def standardize_fit(
    train: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    mean = np.mean(
        train,
        axis=0,
    )

    scale = np.std(
        train,
        axis=0,
        ddof=0,
    )

    scale = np.where(
        scale > 1e-12,
        scale,
        1.0,
    )

    return (
        mean,
        scale,
    )


def transform(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (
        values
        - mean
    ) / scale

    return result.astype(
        np.float32
    )


def log_loss(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    probability = np.clip(
        probability,
        1e-12,
        1.0 - 1e-12,
    )

    return float(
        -np.mean(
            target
            * np.log(
                probability
            )
            + (
                1.0
                - target
            )
            * np.log(
                1.0
                - probability
            )
        )
    )


def brier(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    return float(
        np.mean(
            (
                probability
                - target
            ) ** 2
        )
    )


def accuracy(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    prediction = (
        probability >= 0.5
    ).astype(
        np.float32
    )

    return float(
        np.mean(
            prediction
            == target
        )
    )


def metric_set(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "log_loss":
            log_loss(
                target,
                probability,
            ),
        "brier":
            brier(
                target,
                probability,
            ),
        "accuracy":
            accuracy(
                target,
                probability,
            ),
    }


def train_epoch(
    *,
    model,
    optimizer,
    price_x: torch.Tensor,
    flow_x: torch.Tensor | None,
    target: torch.Tensor,
    generator: torch.Generator,
) -> None:
    permutation = torch.randperm(
        price_x.shape[0],
        generator=generator,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    for start in range(
        0,
        price_x.shape[0],
        BATCH_SIZE,
    ):
        indices = permutation[
            start:
            start + BATCH_SIZE
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        if flow_x is None:
            logits = model(
                price_x[
                    indices
                ]
            )
        else:
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

        optimizer.step()


def probabilities(
    model,
    price_x: torch.Tensor,
    flow_x: torch.Tensor | None,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        if flow_x is None:
            logits = model(
                price_x
            )
        else:
            logits = model(
                price_x,
                flow_x,
            )

        values = torch.sigmoid(
            logits
        )

    return np.asarray(
        values
        .cpu()
        .numpy(),
        dtype=np.float64,
    )


def prepare_arrays(
    data,
    *,
    include_flow: bool,
):
    train = data.train_slice
    validation = data.validation_slice
    test = data.test_slice

    price_mean, price_scale = (
        standardize_fit(
            data.price_x[
                train
            ]
        )
    )

    price_train = transform(
        data.price_x[
            train
        ],
        price_mean,
        price_scale,
    )

    price_validation = transform(
        data.price_x[
            validation
        ],
        price_mean,
        price_scale,
    )

    price_test = transform(
        data.price_x[
            test
        ],
        price_mean,
        price_scale,
    )

    if include_flow:
        flow_mean, flow_scale = (
            standardize_fit(
                data.flow_x[
                    train
                ]
            )
        )

        flow_train = transform(
            data.flow_x[
                train
            ],
            flow_mean,
            flow_scale,
        )

        flow_validation = transform(
            data.flow_x[
                validation
            ],
            flow_mean,
            flow_scale,
        )

        flow_test = transform(
            data.flow_x[
                test
            ],
            flow_mean,
            flow_scale,
        )

    else:
        flow_train = None
        flow_validation = None
        flow_test = None

    return (
        price_train,
        price_validation,
        price_test,
        flow_train,
        flow_validation,
        flow_test,
        data.target[
            train
        ],
        data.target[
            validation
        ],
        data.target[
            test
        ],
    )


def select_epochs(
    data,
    *,
    model_type: str,
    seed: int,
) -> int:
    include_flow = (
        model_type
        == "multimodal"
    )

    (
        price_train,
        price_validation,
        _,
        flow_train,
        flow_validation,
        _,
        target_train,
        target_validation,
        _,
    ) = prepare_arrays(
        data,
        include_flow=
            include_flow,
    )

    set_research_seed(
        seed
    )

    if include_flow:
        model = (
            MultimodalLatentModel()
        )
    else:
        model = (
            PriceLatentModel()
        )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 100000
    )

    price_train_t = torch.tensor(
        price_train,
        dtype=torch.float32,
    )

    price_validation_t = torch.tensor(
        price_validation,
        dtype=torch.float32,
    )

    target_train_t = torch.tensor(
        target_train,
        dtype=torch.float32,
    )

    if include_flow:
        flow_train_t = torch.tensor(
            flow_train,
            dtype=torch.float32,
        )

        flow_validation_t = torch.tensor(
            flow_validation,
            dtype=torch.float32,
        )

    else:
        flow_train_t = None
        flow_validation_t = None

    best_epoch = 1
    best_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        train_epoch(
            model=model,
            optimizer=optimizer,
            price_x=price_train_t,
            flow_x=flow_train_t,
            target=target_train_t,
            generator=generator,
        )

        validation_probability = probabilities(
            model,
            price_validation_t,
            flow_validation_t,
        )

        validation_loss = log_loss(
            target_validation,
            validation_probability,
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

    return best_epoch


def refit_and_predict(
    data,
    *,
    model_type: str,
    epochs: int,
    seed: int,
) -> np.ndarray:
    include_flow = (
        model_type
        == "multimodal"
    )

    development_end = (
        data.validation_slice.stop
    )

    development_price = (
        data.price_x[
            :development_end
        ]
    )

    price_mean, price_scale = (
        standardize_fit(
            development_price
        )
    )

    price_train = transform(
        development_price,
        price_mean,
        price_scale,
    )

    price_test = transform(
        data.price_x[
            data.test_slice
        ],
        price_mean,
        price_scale,
    )

    if include_flow:
        development_flow = (
            data.flow_x[
                :development_end
            ]
        )

        flow_mean, flow_scale = (
            standardize_fit(
                development_flow
            )
        )

        flow_train = transform(
            development_flow,
            flow_mean,
            flow_scale,
        )

        flow_test = transform(
            data.flow_x[
                data.test_slice
            ],
            flow_mean,
            flow_scale,
        )

    else:
        flow_train = None
        flow_test = None

    target_train = (
        data.target[
            :development_end
        ]
    )

    set_research_seed(
        seed
    )

    if include_flow:
        model = (
            MultimodalLatentModel()
        )
    else:
        model = (
            PriceLatentModel()
        )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 200000
    )

    price_train_t = torch.tensor(
        price_train,
        dtype=torch.float32,
    )

    target_train_t = torch.tensor(
        target_train,
        dtype=torch.float32,
    )

    if include_flow:
        flow_train_t = torch.tensor(
            flow_train,
            dtype=torch.float32,
        )
    else:
        flow_train_t = None

    for _ in range(
        epochs
    ):
        model.train()

        train_epoch(
            model=model,
            optimizer=optimizer,
            price_x=price_train_t,
            flow_x=flow_train_t,
            target=target_train_t,
            generator=generator,
        )

    price_test_t = torch.tensor(
        price_test,
        dtype=torch.float32,
    )

    if include_flow:
        flow_test_t = torch.tensor(
            flow_test,
            dtype=torch.float32,
        )
    else:
        flow_test_t = None

    return probabilities(
        model,
        price_test_t,
        flow_test_t,
    )


def paired_summary(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    mean_value = float(
        np.mean(
            array
        )
    )

    standard_error = float(
        np.std(
            array,
            ddof=1,
        )
        / np.sqrt(
            array.size
        )
    )

    statistic = (
        mean_value
        / standard_error
        if standard_error > 0.0
        else 0.0
    )

    p_value = math.erfc(
        abs(
            statistic
        )
        / math.sqrt(
            2.0
        )
    )

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
                    array < 0.0
                )
            ),
        "observations":
            int(
                array.size
            ),
    }


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    run_rows = []

    for repetition in range(
        REPETITIONS
    ):
        seed = (
            BASE_SEED
            + repetition
        )

        reference_data = (
            generate_synthetic_latent_dataset(
                SCENARIOS[
                    "stationary_gaussian"
                ],
                seed=seed,
            )
        )

        price_selection_seed = (
            seed
            + 1000000
        )

        price_refit_seed = (
            seed
            + 2000000
        )

        price_epoch = select_epochs(
            reference_data,
            model_type="price",
            seed=
                price_selection_seed,
        )

        price_probability = (
            refit_and_predict(
                reference_data,
                model_type="price",
                epochs=
                    price_epoch,
                seed=
                    price_refit_seed,
            )
        )

        price_metrics = metric_set(
            reference_data.target[
                reference_data.test_slice
            ],
            price_probability,
        )

        for (
            scenario_index,
            (
                scenario_name,
                config,
            ),
        ) in enumerate(
            SCENARIOS.items()
        ):
            data = (
                generate_synthetic_latent_dataset(
                    config,
                    seed=seed,
                )
            )

            if not np.array_equal(
                data.target,
                reference_data.target,
            ):
                raise RuntimeError(
                    "Scenario target mismatch."
                )

            if not np.array_equal(
                data.price_x,
                reference_data.price_x,
            ):
                raise RuntimeError(
                    "Scenario Price data mismatch."
                )

            selection_seed = (
                seed
                + 3000000
                + scenario_index
                * 10000
            )

            refit_seed = (
                seed
                + 4000000
                + scenario_index
                * 10000
            )

            multimodal_epoch = (
                select_epochs(
                    data,
                    model_type=
                        "multimodal",
                    seed=
                        selection_seed,
                )
            )

            multimodal_probability = (
                refit_and_predict(
                    data,
                    model_type=
                        "multimodal",
                    epochs=
                        multimodal_epoch,
                    seed=
                        refit_seed,
                )
            )

            target_test = (
                data.target[
                    data.test_slice
                ]
            )

            multimodal_metrics = (
                metric_set(
                    target_test,
                    multimodal_probability,
                )
            )

            oracle_metrics = (
                metric_set(
                    target_test,
                    data.true_probability[
                        data.test_slice
                    ],
                )
            )

            row = {
                "repetition":
                    repetition,
                "seed":
                    seed,
                "scenario":
                    scenario_name,
                "price_epoch":
                    price_epoch,
                "multimodal_epoch":
                    multimodal_epoch,
                "oracle_log_loss":
                    oracle_metrics[
                        "log_loss"
                    ],
                "price_log_loss":
                    price_metrics[
                        "log_loss"
                    ],
                "multimodal_log_loss":
                    multimodal_metrics[
                        "log_loss"
                    ],
                "multimodal_minus_price":
                    (
                        multimodal_metrics[
                            "log_loss"
                        ]
                        - price_metrics[
                            "log_loss"
                        ]
                    ),
                "price_brier":
                    price_metrics[
                        "brier"
                    ],
                "multimodal_brier":
                    multimodal_metrics[
                        "brier"
                    ],
                "price_accuracy":
                    price_metrics[
                        "accuracy"
                    ],
                "multimodal_accuracy":
                    multimodal_metrics[
                        "accuracy"
                    ],
            }

            run_rows.append(
                row
            )

            print(
                f"rep={repetition:02d} "
                f"{scenario_name:30s} "
                f"Price="
                f"{price_metrics['log_loss']:.5f} "
                f"Multi="
                f"{multimodal_metrics['log_loss']:.5f} "
                f"delta="
                f"{row['multimodal_minus_price']:+.5f} "
                f"epochs="
                f"{price_epoch:02d}/"
                f"{multimodal_epoch:02d}"
            )

    summaries = {}

    print()
    print(
        "SYNTHETIC LATENT BENCHMARK"
    )

    print(
        "=========================="
    )

    for scenario_name in (
        SCENARIOS
    ):
        rows = [
            row
            for row in run_rows
            if row[
                "scenario"
            ]
            == scenario_name
        ]

        price_ll = np.asarray(
            [
                row[
                    "price_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        multi_ll = np.asarray(
            [
                row[
                    "multimodal_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        oracle_ll = np.asarray(
            [
                row[
                    "oracle_log_loss"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        delta = (
            multi_ll
            - price_ll
        )

        comparison = (
            paired_summary(
                delta.tolist()
            )
        )

        summaries[
            scenario_name
        ] = {
            "mean_oracle_log_loss":
                float(
                    np.mean(
                        oracle_ll
                    )
                ),
            "mean_price_log_loss":
                float(
                    np.mean(
                        price_ll
                    )
                ),
            "mean_multimodal_log_loss":
                float(
                    np.mean(
                        multi_ll
                    )
                ),
            "multimodal_minus_price":
                comparison,
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
            "Oracle LL:",
            f"{np.mean(oracle_ll):.6f}",
        )

        print(
            "Price LL:",
            f"{np.mean(price_ll):.6f}",
        )

        print(
            "Multi LL:",
            f"{np.mean(multi_ll):.6f}",
        )

        print(
            "Multi - Price:",
            f"{comparison['mean']:+.6f}",
        )

        print(
            "SE:",
            f"{comparison['standard_error']:.6f}",
        )

        print(
            "t:",
            f"{comparison['t_statistic']:+.3f}",
        )

        print(
            "normal p:",
            f"{comparison['normal_p_value']:.6f}",
        )

        print(
            "Multimodal wins:",
            f"{comparison['wins']}/"
            f"{comparison['observations']}",
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
                run_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            run_rows
        )

    results = {
        "experiment":
            "synthetic-latent-benchmark",
        "status":
            (
                "Developmental controlled benchmark."
            ),
        "repetitions":
            REPETITIONS,
        "scenarios": {
            name: {
                "flow_heavy_tail_df":
                    config
                    .flow_heavy_tail_df,
                "flow_test_loading_multiplier":
                    config
                    .flow_test_loading_multiplier,
                "flow_test_noise_multiplier":
                    config
                    .flow_test_noise_multiplier,
            }
            for name, config
            in SCENARIOS.items()
        },
        "summaries":
            summaries,
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
