from __future__ import annotations

import csv
import json
import math

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import erf, sqrt
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from latent_information_dynamics.models.latent import (
    FeatureStandardizer,
    MultimodalLatentModel,
    binary_training_step,
    set_research_seed,
)


SUPERVISED_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_4h_2020_2025.csv"
)

SUPERVISED_METADATA_FILE = Path(
    "data/derived/supervised/"
    "btc_combined_supervised_4h_2020_2025.json"
)

BASELINE_FILE = Path(
    "data/derived/experiments/"
    "baseline_information_decomposition/"
    "baseline_model_predictions_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/walk_forward"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "latent_neural_walk_forward_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "latent_neural_walk_forward_predictions.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "latent_neural_walk_forward_gradients.csv"
)

EPOCH_FILE = (
    OUTPUT_ROOT
    / "latent_neural_walk_forward_epochs.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "latent_neural_walk_forward_folds.csv"
)


PRICE_FEATURES = (
    "return_5m",
    "return_30m",
    "return_1h",
    "return_4h",
    "return_12h",
    "return_24h",
    "rv_1h",
    "rv_4h",
    "rv_24h",
    "range_4h",
    "range_24h",
)

SEED = 20261016

EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HAC_LAGS = (
    0,
    1,
    6,
    12,
    42,
)

EPSILON = 1e-15


@dataclass(
    frozen=True,
    slots=True,
)
class Example:
    decision_time: datetime

    label_available_time: datetime

    target: int

    price_x: tuple[float, ...]

    flow_x: tuple[float, ...]


def parse_datetime(
    value: str,
) -> datetime:
    return datetime.fromisoformat(
        value
    )


def observation_log_loss(
    target: int,
    probability: float,
) -> float:
    probability = min(
        max(
            probability,
            EPSILON,
        ),
        1.0 - EPSILON,
    )

    return -(
        target
        * math.log(
            probability
        )
        + (
            1 - target
        )
        * math.log(
            1.0 - probability
        )
    )


def observation_brier(
    target: int,
    probability: float,
) -> float:
    return (
        probability - target
    ) ** 2


def observation_accuracy(
    target: int,
    probability: float,
) -> int:
    return int(
        (
            probability >= 0.5
        )
        == target
    )


def metrics(
    rows: list[dict[str, object]],
    probability_column: str,
) -> dict[str, float]:
    probabilities = [
        float(
            row[
                probability_column
            ]
        )
        for row
        in rows
    ]

    targets = [
        int(
            row[
                "target"
            ]
        )
        for row
        in rows
    ]

    return {
        "log_loss":
            mean(
                observation_log_loss(
                    target,
                    probability,
                )
                for (
                    target,
                    probability,
                )
                in zip(
                    targets,
                    probabilities,
                    strict=True,
                )
            ),
        "brier_score":
            mean(
                observation_brier(
                    target,
                    probability,
                )
                for (
                    target,
                    probability,
                )
                in zip(
                    targets,
                    probabilities,
                    strict=True,
                )
            ),
        "accuracy":
            mean(
                observation_accuracy(
                    target,
                    probability,
                )
                for (
                    target,
                    probability,
                )
                in zip(
                    targets,
                    probabilities,
                    strict=True,
                )
            ),
        "mean_probability":
            mean(
                probabilities
            ),
    }


def percentile(
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


def gradient_summary(
    rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    total = [
        float(
            row[
                "total_norm"
            ]
        )
        for row
        in rows
    ]

    price = [
        float(
            row[
                "price_encoder_norm"
            ]
        )
        for row
        in rows
    ]

    flow = [
        float(
            row[
                "flow_encoder_norm"
            ]
        )
        for row
        in rows
    ]

    fusion = [
        float(
            row[
                "fusion_norm"
            ]
        )
        for row
        in rows
    ]

    head = [
        float(
            row[
                "prediction_head_norm"
            ]
        )
        for row
        in rows
    ]

    maximum = [
        float(
            row[
                "max_abs_gradient"
            ]
        )
        for row
        in rows
    ]

    ratios = [
        flow_value
        / max(
            price_value,
            1e-15,
        )
        for (
            price_value,
            flow_value,
        )
        in zip(
            price,
            flow,
            strict=True,
        )
    ]

    def summary(
        values: list[float],
    ) -> dict[str, float]:
        return {
            "mean":
                mean(
                    values
                ),
            "median":
                percentile(
                    values,
                    0.50,
                ),
            "p90":
                percentile(
                    values,
                    0.90,
                ),
            "p95":
                percentile(
                    values,
                    0.95,
                ),
            "p99":
                percentile(
                    values,
                    0.99,
                ),
            "maximum":
                max(
                    values
                ),
        }

    return {
        "steps":
            len(
                rows
            ),
        "total_norm":
            summary(
                total
            ),
        "price_encoder_norm":
            summary(
                price
            ),
        "flow_encoder_norm":
            summary(
                flow
            ),
        "fusion_norm":
            summary(
                fusion
            ),
        "prediction_head_norm":
            summary(
                head
            ),
        "max_abs_gradient":
            summary(
                maximum
            ),
        "flow_to_price_gradient_ratio":
            summary(
                ratios
            ),
        "nonfinite_gradient_elements":
            sum(
                int(
                    row[
                        "nonfinite_gradient_elements"
                    ]
                )
                for row
                in rows
            ),
    }


def hac_standard_error(
    values: list[float],
    lag: int,
) -> float:
    n = len(
        values
    )

    mu = mean(
        values
    )

    centered = [
        value - mu
        for value
        in values
    ]

    gamma_zero = (
        sum(
            value * value
            for value
            in centered
        )
        / n
    )

    long_run_variance = (
        gamma_zero
    )

    for ell in range(
        1,
        lag + 1,
    ):
        covariance = (
            sum(
                centered[t]
                * centered[
                    t - ell
                ]
                for t
                in range(
                    ell,
                    n,
                )
            )
            / n
        )

        weight = (
            1.0
            - ell
            / (
                lag + 1.0
            )
        )

        long_run_variance += (
            2.0
            * weight
            * covariance
        )

    return math.sqrt(
        max(
            long_run_variance,
            0.0,
        )
        / n
    )


def two_sided_normal_p(
    statistic: float,
) -> float:
    z = abs(
        statistic
    )

    cdf = (
        0.5
        * (
            1.0
            + erf(
                z
                / sqrt(
                    2.0
                )
            )
        )
    )

    return (
        2.0
        * (
            1.0
            - cdf
        )
    )


def hac_comparison(
    rows: list[dict[str, object]],
    first_column: str,
    second_column: str,
) -> dict[str, object]:
    log_loss_difference = []

    brier_difference = []

    for row in rows:
        target = int(
            row[
                "target"
            ]
        )

        first = float(
            row[
                first_column
            ]
        )

        second = float(
            row[
                second_column
            ]
        )

        log_loss_difference.append(
            observation_log_loss(
                target,
                second,
            )
            - observation_log_loss(
                target,
                first,
            )
        )

        brier_difference.append(
            observation_brier(
                target,
                second,
            )
            - observation_brier(
                target,
                first,
            )
        )

    result = {}

    for name, values in (
        (
            "log_loss",
            log_loss_difference,
        ),
        (
            "brier",
            brier_difference,
        ),
    ):
        difference_mean = mean(
            values
        )

        hac = {}

        for lag in HAC_LAGS:
            standard_error = (
                hac_standard_error(
                    values,
                    lag,
                )
            )

            statistic = (
                difference_mean
                / standard_error
                if standard_error > 0.0
                else 0.0
            )

            hac[
                str(
                    lag
                )
            ] = {
                "standard_error":
                    standard_error,
                "t_statistic":
                    statistic,
                "p_value":
                    two_sided_normal_p(
                        statistic
                    ),
            }

        result[
            name
        ] = {
            "mean_difference":
                difference_mean,
            "hac":
                hac,
        }

    return result


def load_examples() -> tuple[
    list[Example],
    tuple[str, ...],
]:
    metadata = json.loads(
        SUPERVISED_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    feature_names = tuple(
        metadata[
            "features"
        ]
    )

    if len(
        feature_names
    ) != 31:
        raise RuntimeError(
            "Expected 31 combined features."
        )

    missing_price = (
        set(
            PRICE_FEATURES
        )
        - set(
            feature_names
        )
    )

    if missing_price:
        raise RuntimeError(
            "Missing price features: "
            f"{sorted(missing_price)}"
        )

    flow_features = tuple(
        feature
        for feature
        in feature_names
        if feature
        not in PRICE_FEATURES
    )

    if len(
        flow_features
    ) != 20:
        raise RuntimeError(
            "Expected 20 flow features."
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
                Example(
                    decision_time=
                        parse_datetime(
                            raw[
                                "decision_time"
                            ]
                        ),
                    label_available_time=
                        parse_datetime(
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
                                    feature
                                ]
                            )
                            for feature
                            in PRICE_FEATURES
                        ),
                    flow_x=
                        tuple(
                            float(
                                raw[
                                    feature
                                ]
                            )
                            for feature
                            in flow_features
                        ),
                )
            )

    if not examples:
        raise RuntimeError(
            "No supervised examples."
        )

    for previous, current in zip(
        examples,
        examples[1:],
        strict=False,
    ):
        if (
            current.decision_time
            <= previous.decision_time
        ):
            raise RuntimeError(
                "Supervised examples are "
                "not strictly chronological."
            )

    return (
        examples,
        flow_features,
    )


def load_baseline_rows() -> list[
    dict[str, object]
]:
    rows = []

    with BASELINE_FILE.open(
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
            "mprice_probability",
            "mcombined_probability",
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
                "Frozen baseline is missing "
                f"columns: {sorted(missing)}"
            )

        for raw in reader:
            rows.append(
                {
                    "fold":
                        int(
                            raw[
                                "fold"
                            ]
                        ),
                    "decision_time":
                        parse_datetime(
                            raw[
                                "decision_time"
                            ]
                        ),
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
                    "mprice_probability":
                        float(
                            raw[
                                "mprice_probability"
                            ]
                        ),
                    "mcombined_probability":
                        float(
                            raw[
                                "mcombined_probability"
                            ]
                        ),
                }
            )

    if len(
        rows
    ) != 10840:
        raise RuntimeError(
            "Expected 10,840 frozen OOS rows, "
            f"got {len(rows)}."
        )

    return rows


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    (
        examples,
        flow_features,
    ) = load_examples()

    example_by_time = {
        example.decision_time:
            example
        for example
        in examples
    }

    baseline_rows = (
        load_baseline_rows()
    )

    grouped = defaultdict(
        list
    )

    for row in baseline_rows:
        grouped[
            int(
                row[
                    "fold"
                ]
            )
        ].append(
            row
        )

    folds = sorted(
        grouped
    )

    if folds != list(
        range(
            20
        )
    ):
        raise RuntimeError(
            "Expected folds 0 through 19."
        )

    all_prediction_rows = []

    all_gradient_rows = []

    all_epoch_rows = []

    fold_json = []

    fold_csv = []

    global_gradient_step = 0

    previous_train_count = 0

    print(
        "LATENT NEURAL WALK-FORWARD"
    )

    print(
        "=========================="
    )

    for fold in folds:
        frozen_rows = grouped[
            fold
        ]

        frozen_rows.sort(
            key=lambda row:
                row[
                    "decision_time"
                ]
        )

        test_start = (
            frozen_rows[
                0
            ][
                "decision_time"
            ]
        )

        train_examples = [
            example
            for example
            in examples
            if (
                example.decision_time
                < test_start
                and
                example.label_available_time
                <= test_start
            )
        ]

        if (
            len(
                train_examples
            )
            <= previous_train_count
            and fold > 0
        ):
            raise RuntimeError(
                "Expanding training sample "
                "did not grow."
            )

        previous_train_count = len(
            train_examples
        )

        test_examples = []

        for frozen in frozen_rows:
            timestamp = frozen[
                "decision_time"
            ]

            if timestamp not in example_by_time:
                raise RuntimeError(
                    "Frozen OOS timestamp missing "
                    "from supervised dataset."
                )

            example = (
                example_by_time[
                    timestamp
                ]
            )

            if (
                example.target
                != int(
                    frozen[
                        "target"
                    ]
                )
            ):
                raise RuntimeError(
                    "Target mismatch with "
                    "frozen baseline."
                )

            test_examples.append(
                example
            )

        if any(
            example.label_available_time
            > test_start
            for example
            in train_examples
        ):
            raise RuntimeError(
                "PIT label availability failure."
            )

        train_times = {
            example.decision_time
            for example
            in train_examples
        }

        test_times = {
            example.decision_time
            for example
            in test_examples
        }

        if (
            train_times
            & test_times
        ):
            raise RuntimeError(
                "Train/test timestamp overlap."
            )

        price_standardizer = (
            FeatureStandardizer.fit(
                [
                    example.price_x
                    for example
                    in train_examples
                ]
            )
        )

        flow_standardizer = (
            FeatureStandardizer.fit(
                [
                    example.flow_x
                    for example
                    in train_examples
                ]
            )
        )

        price_train = torch.from_numpy(
            price_standardizer.transform(
                [
                    example.price_x
                    for example
                    in train_examples
                ]
            )
        )

        flow_train = torch.from_numpy(
            flow_standardizer.transform(
                [
                    example.flow_x
                    for example
                    in train_examples
                ]
            )
        )

        target_train = torch.tensor(
            [
                example.target
                for example
                in train_examples
            ],
            dtype=torch.float32,
        )

        price_test = torch.from_numpy(
            price_standardizer.transform(
                [
                    example.price_x
                    for example
                    in test_examples
                ]
            )
        )

        flow_test = torch.from_numpy(
            flow_standardizer.transform(
                [
                    example.flow_x
                    for example
                    in test_examples
                ]
            )
        )

        fold_seed = (
            SEED + fold
        )

        set_research_seed(
            fold_seed
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

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            SEED
            + 100000
            + fold
        )

        model.train()

        fold_gradient_rows = []

        fold_epoch_rows = []

        fold_step = 0

        for epoch in range(
            EPOCHS
        ):
            permutation = (
                torch.randperm(
                    len(
                        train_examples
                    ),
                    generator=
                        generator,
                )
            )

            weighted_loss = 0.0

            observations = 0

            epoch_gradient_norms = []

            for (
                batch_number,
                start,
            ) in enumerate(
                range(
                    0,
                    len(
                        train_examples
                    ),
                    BATCH_SIZE,
                )
            ):
                indices = permutation[
                    start:
                    start
                    + BATCH_SIZE
                ]

                result = (
                    binary_training_step(
                        model=
                            model,
                        optimizer=
                            optimizer,
                        price_x=
                            price_train[
                                indices
                            ],
                        flow_x=
                            flow_train[
                                indices
                            ],
                        target=
                            target_train[
                                indices
                            ],
                    )
                )

                batch_observations = int(
                    indices.shape[
                        0
                    ]
                )

                weighted_loss += (
                    result.loss
                    * batch_observations
                )

                observations += (
                    batch_observations
                )

                snapshot = (
                    result.gradients
                )

                gradient_row = {
                    "global_step":
                        global_gradient_step,
                    "fold":
                        fold,
                    "fold_step":
                        fold_step,
                    "epoch":
                        epoch,
                    "batch":
                        batch_number,
                    "batch_observations":
                        batch_observations,
                    "loss":
                        result.loss,
                    **snapshot.as_dict(),
                }

                fold_gradient_rows.append(
                    gradient_row
                )

                all_gradient_rows.append(
                    gradient_row
                )

                epoch_gradient_norms.append(
                    snapshot.total_norm
                )

                global_gradient_step += 1

                fold_step += 1

            epoch_row = {
                "fold":
                    fold,
                "epoch":
                    epoch,
                "mean_training_loss":
                    (
                        weighted_loss
                        / observations
                    ),
                "mean_total_gradient_norm":
                    mean(
                        epoch_gradient_norms
                    ),
                "max_total_gradient_norm":
                    max(
                        epoch_gradient_norms
                    ),
            }

            fold_epoch_rows.append(
                epoch_row
            )

            all_epoch_rows.append(
                epoch_row
            )

        model.eval()

        with torch.no_grad():
            output = (
                model.forward_with_state(
                    price_test,
                    flow_test,
                )
            )

            probabilities = (
                torch.sigmoid(
                    output.logits
                )
                .cpu()
                .tolist()
            )

        price_states = (
            output.price_state
            .cpu()
            .numpy()
        )

        flow_states = (
            output.flow_state
            .cpu()
            .numpy()
        )

        latent_states = (
            output.latent_state
            .cpu()
            .numpy()
        )

        fold_prediction_rows = []

        for index, (
            frozen,
            example,
            probability,
        ) in enumerate(
            zip(
                frozen_rows,
                test_examples,
                probabilities,
                strict=True,
            )
        ):
            prediction = {
                "fold":
                    fold,
                "decision_time":
                    example
                    .decision_time
                    .isoformat(),
                "target":
                    example.target,
                "m0_probability":
                    float(
                        frozen[
                            "m0_probability"
                        ]
                    ),
                "mprice_probability":
                    float(
                        frozen[
                            "mprice_probability"
                        ]
                    ),
                "mcombined_probability":
                    float(
                        frozen[
                            "mcombined_probability"
                        ]
                    ),
                "mlatent_probability":
                    float(
                        probability
                    ),
            }

            for state_index in range(
                price_states.shape[
                    1
                ]
            ):
                prediction[
                    f"price_state_{state_index}"
                ] = float(
                    price_states[
                        index,
                        state_index,
                    ]
                )

            for state_index in range(
                flow_states.shape[
                    1
                ]
            ):
                prediction[
                    f"flow_state_{state_index}"
                ] = float(
                    flow_states[
                        index,
                        state_index,
                    ]
                )

            for state_index in range(
                latent_states.shape[
                    1
                ]
            ):
                prediction[
                    f"latent_state_{state_index}"
                ] = float(
                    latent_states[
                        index,
                        state_index,
                    ]
                )

            fold_prediction_rows.append(
                prediction
            )

            all_prediction_rows.append(
                prediction
            )

        price_metrics = metrics(
            fold_prediction_rows,
            "mprice_probability",
        )

        combined_metrics = metrics(
            fold_prediction_rows,
            "mcombined_probability",
        )

        latent_metrics = metrics(
            fold_prediction_rows,
            "mlatent_probability",
        )

        m0_metrics = metrics(
            fold_prediction_rows,
            "m0_probability",
        )

        gradients = (
            gradient_summary(
                fold_gradient_rows
            )
        )

        fold_record = {
            "fold":
                fold,
            "seed":
                fold_seed,
            "test_start":
                test_start.isoformat(),
            "train_observations":
                len(
                    train_examples
                ),
            "test_observations":
                len(
                    test_examples
                ),
            "standardization": {
                "price":
                    price_standardizer
                    .as_dict(),
                "flow":
                    flow_standardizer
                    .as_dict(),
            },
            "metrics": {
                "M0":
                    m0_metrics,
                "M_PRICE":
                    price_metrics,
                "M_COMBINED_RAW":
                    combined_metrics,
                "M_LATENT":
                    latent_metrics,
            },
            "latent_minus_price": {
                "log_loss":
                    (
                        latent_metrics[
                            "log_loss"
                        ]
                        - price_metrics[
                            "log_loss"
                        ]
                    ),
                "brier":
                    (
                        latent_metrics[
                            "brier_score"
                        ]
                        - price_metrics[
                            "brier_score"
                        ]
                    ),
                "accuracy":
                    (
                        latent_metrics[
                            "accuracy"
                        ]
                        - price_metrics[
                            "accuracy"
                        ]
                    ),
            },
            "gradient_diagnostics":
                gradients,
            "training": {
                "first_epoch":
                    fold_epoch_rows[
                        0
                    ],
                "last_epoch":
                    fold_epoch_rows[
                        -1
                    ],
            },
        }

        fold_json.append(
            fold_record
        )

        fold_csv.append(
            {
                "fold":
                    fold,
                "seed":
                    fold_seed,
                "test_start":
                    test_start
                    .isoformat(),
                "train_observations":
                    len(
                        train_examples
                    ),
                "test_observations":
                    len(
                        test_examples
                    ),
                "m0_log_loss":
                    m0_metrics[
                        "log_loss"
                    ],
                "price_log_loss":
                    price_metrics[
                        "log_loss"
                    ],
                "combined_log_loss":
                    combined_metrics[
                        "log_loss"
                    ],
                "latent_log_loss":
                    latent_metrics[
                        "log_loss"
                    ],
                "latent_minus_price_log_loss":
                    (
                        latent_metrics[
                            "log_loss"
                        ]
                        - price_metrics[
                            "log_loss"
                        ]
                    ),
                "price_brier":
                    price_metrics[
                        "brier_score"
                    ],
                "latent_brier":
                    latent_metrics[
                        "brier_score"
                    ],
                "latent_minus_price_brier":
                    (
                        latent_metrics[
                            "brier_score"
                        ]
                        - price_metrics[
                            "brier_score"
                        ]
                    ),
                "price_accuracy":
                    price_metrics[
                        "accuracy"
                    ],
                "latent_accuracy":
                    latent_metrics[
                        "accuracy"
                    ],
                "latent_minus_price_accuracy":
                    (
                        latent_metrics[
                            "accuracy"
                        ]
                        - price_metrics[
                            "accuracy"
                        ]
                    ),
                "mean_total_gradient_norm":
                    gradients[
                        "total_norm"
                    ][
                        "mean"
                    ],
                "p95_total_gradient_norm":
                    gradients[
                        "total_norm"
                    ][
                        "p95"
                    ],
                "mean_price_gradient_norm":
                    gradients[
                        "price_encoder_norm"
                    ][
                        "mean"
                    ],
                "mean_flow_gradient_norm":
                    gradients[
                        "flow_encoder_norm"
                    ][
                        "mean"
                    ],
                "mean_flow_to_price_ratio":
                    gradients[
                        "flow_to_price_gradient_ratio"
                    ][
                        "mean"
                    ],
                "nonfinite_gradient_elements":
                    gradients[
                        "nonfinite_gradient_elements"
                    ],
            }
        )

        print(
            f"Fold {fold:02d}: "
            f"train={len(train_examples):5d} "
            f"test={len(test_examples):3d} "
            f"Price LL="
            f"{price_metrics['log_loss']:.6f} "
            f"Latent LL="
            f"{latent_metrics['log_loss']:.6f} "
            f"delta="
            f"{latent_metrics['log_loss'] - price_metrics['log_loss']:+.6f} "
            f"flow/price grad="
            f"{gradients['flow_to_price_gradient_ratio']['mean']:.3f}"
        )

    aggregate = {
        "M0":
            metrics(
                all_prediction_rows,
                "m0_probability",
            ),
        "M_PRICE":
            metrics(
                all_prediction_rows,
                "mprice_probability",
            ),
        "M_COMBINED_RAW":
            metrics(
                all_prediction_rows,
                "mcombined_probability",
            ),
        "M_LATENT":
            metrics(
                all_prediction_rows,
                "mlatent_probability",
            ),
    }

    price_comparison = (
        hac_comparison(
            all_prediction_rows,
            "mprice_probability",
            "mlatent_probability",
        )
    )

    m0_comparison = (
        hac_comparison(
            all_prediction_rows,
            "m0_probability",
            "mlatent_probability",
        )
    )

    combined_comparison = (
        hac_comparison(
            all_prediction_rows,
            "mcombined_probability",
            "mlatent_probability",
        )
    )

    fold_wins = {
        "log_loss":
            sum(
                record[
                    "latent_minus_price"
                ][
                    "log_loss"
                ] < 0.0
                for record
                in fold_json
            ),
        "brier":
            sum(
                record[
                    "latent_minus_price"
                ][
                    "brier"
                ] < 0.0
                for record
                in fold_json
            ),
        "accuracy":
            sum(
                record[
                    "latent_minus_price"
                ][
                    "accuracy"
                ] > 0.0
                for record
                in fold_json
            ),
    }

    aggregate_gradients = (
        gradient_summary(
            all_gradient_rows
        )
    )

    results = {
        "experiment":
            "latent-neural-walk-forward-4h",
        "purpose":
            (
                "Fixed-architecture nonlinear multimodal "
                "latent-state benchmark with optimization "
                "instrumentation."
            ),
        "protocol": {
            "outer_folds":
                len(
                    folds
                ),
            "outer_oos_forecasts":
                len(
                    all_prediction_rows
                ),
            "point_in_time":
                True,
            "standardization":
                "fit separately on each legal outer training fold",
            "device":
                "cpu",
            "base_seed":
                SEED,
            "epochs":
                EPOCHS,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "optimizer":
                "AdamW",
            "hyperparameter_search":
                False,
        },
        "architecture": {
            "price":
                "11 -> 16 -> 4",
            "flow":
                "20 -> 24 -> 4",
            "fusion":
                "8 -> 8 -> 4",
            "prediction":
                "4 -> 1 logit",
        },
        "price_feature_count":
            len(
                PRICE_FEATURES
            ),
        "flow_feature_count":
            len(
                flow_features
            ),
        "metrics":
            aggregate,
        "primary_comparison": {
            "first":
                "M_PRICE",
            "second":
                "M_LATENT",
            "difference_definition":
                "second_minus_first",
            "log_loss_delta":
                (
                    aggregate[
                        "M_LATENT"
                    ][
                        "log_loss"
                    ]
                    - aggregate[
                        "M_PRICE"
                    ][
                        "log_loss"
                    ]
                ),
            "brier_delta":
                (
                    aggregate[
                        "M_LATENT"
                    ][
                        "brier_score"
                    ]
                    - aggregate[
                        "M_PRICE"
                    ][
                        "brier_score"
                    ]
                ),
            "accuracy_delta":
                (
                    aggregate[
                        "M_LATENT"
                    ][
                        "accuracy"
                    ]
                    - aggregate[
                        "M_PRICE"
                    ][
                        "accuracy"
                    ]
                ),
            "latent_fold_wins":
                fold_wins,
            "hac":
                price_comparison,
        },
        "comparison_vs_m0":
            m0_comparison,
        "comparison_vs_raw_combined":
            combined_comparison,
        "gradient_diagnostics":
            aggregate_gradients,
        "folds":
            fold_json,
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

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                all_prediction_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_prediction_rows
        )

    with GRADIENT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                all_gradient_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_gradient_rows
        )

    with EPOCH_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                all_epoch_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_epoch_rows
        )

    with FOLD_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fold_csv[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            fold_csv
        )

    print()
    print(
        "AGGREGATE OOS RESULTS"
    )

    print(
        "---------------------"
    )

    for name in (
        "M0",
        "M_PRICE",
        "M_COMBINED_RAW",
        "M_LATENT",
    ):
        item = aggregate[
            name
        ]

        print(
            f"{name:15s} "
            f"LL="
            f"{item['log_loss']:.8f}  "
            f"Brier="
            f"{item['brier_score']:.8f}  "
            f"Acc="
            f"{item['accuracy']:.4%}"
        )

    primary = results[
        "primary_comparison"
    ]

    print()
    print(
        "PRIMARY: LATENT - PRICE"
    )

    print(
        "-----------------------"
    )

    print(
        "Log loss: "
        f"{primary['log_loss_delta']:+.8f}"
    )

    print(
        "Brier:    "
        f"{primary['brier_delta']:+.8f}"
    )

    print(
        "Accuracy: "
        f"{primary['accuracy_delta']:+.4%}"
    )

    print(
        "Latent fold wins: "
        f"LL {fold_wins['log_loss']}/20, "
        f"Brier {fold_wins['brier']}/20, "
        f"Acc {fold_wins['accuracy']}/20"
    )

    print()

    for metric in (
        "log_loss",
        "brier",
    ):
        item = price_comparison[
            metric
        ]

        print(
            f"HAC {metric.upper()} "
            f"mean delta="
            f"{item['mean_difference']:+.8f}"
        )

        for lag in HAC_LAGS:
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
        "AGGREGATE GRADIENTS"
    )

    print(
        "-------------------"
    )

    print(
        "Steps: "
        f"{aggregate_gradients['steps']:,}"
    )

    print(
        "Total mean: "
        f"{aggregate_gradients['total_norm']['mean']:.6f}"
    )

    print(
        "Total p95:  "
        f"{aggregate_gradients['total_norm']['p95']:.6f}"
    )

    print(
        "Total p99:  "
        f"{aggregate_gradients['total_norm']['p99']:.6f}"
    )

    print(
        "Total max:  "
        f"{aggregate_gradients['total_norm']['maximum']:.6f}"
    )

    print(
        "Price mean: "
        f"{aggregate_gradients['price_encoder_norm']['mean']:.6f}"
    )

    print(
        "Flow mean:  "
        f"{aggregate_gradients['flow_encoder_norm']['mean']:.6f}"
    )

    print(
        "Flow/Price mean: "
        f"{aggregate_gradients['flow_to_price_gradient_ratio']['mean']:.6f}"
    )

    print(
        "Flow/Price p95:  "
        f"{aggregate_gradients['flow_to_price_gradient_ratio']['p95']:.6f}"
    )

    print(
        "Non-finite: "
        f"{aggregate_gradients['nonfinite_gradient_elements']}"
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
        "Gradients   ->",
        GRADIENT_FILE,
    )

    print(
        "Epochs      ->",
        EPOCH_FILE,
    )

    print(
        "Folds       ->",
        FOLD_FILE,
    )


if __name__ == "__main__":
    main()
