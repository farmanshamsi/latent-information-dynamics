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
from torch import nn

from latent_information_dynamics.models.latent import (
    FeatureStandardizer,
    MultimodalLatentModel,
    PriceLatentModel,
    parameter_gradient_norm,
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
    "latent_neural/nested_walk_forward"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "nested_latent_neural_walk_forward_predictions.csv"
)

SELECTION_FILE = (
    OUTPUT_ROOT
    / "nested_latent_epoch_selection.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "nested_latent_refit_gradients.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "nested_latent_folds.csv"
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

INNER_VALIDATION_OBSERVATIONS = 546

MAX_EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

BASE_SEED = 20261016

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
    p = min(
        max(
            probability,
            EPSILON,
        ),
        1.0 - EPSILON,
    )

    return -(
        target * math.log(p)
        + (1 - target)
        * math.log(
            1.0 - p
        )
    )


def observation_brier(
    target: int,
    probability: float,
) -> float:
    return (
        probability - target
    ) ** 2


def metric_set(
    targets: list[int],
    probabilities: list[float],
) -> dict[str, float]:
    return {
        "log_loss":
            mean(
                observation_log_loss(
                    target,
                    probability,
                )
                for target, probability
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
                for target, probability
                in zip(
                    targets,
                    probabilities,
                    strict=True,
                )
            ),
        "accuracy":
            mean(
                int(
                    (
                        probability >= 0.5
                    )
                    == target
                )
                for target, probability
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

    flow_features = tuple(
        feature
        for feature in feature_names
        if feature not in PRICE_FEATURES
    )

    if len(PRICE_FEATURES) != 11:
        raise RuntimeError(
            "Expected 11 price features."
        )

    if len(flow_features) != 20:
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
                                raw[name]
                            )
                            for name
                            in PRICE_FEATURES
                        ),
                    flow_x=
                        tuple(
                            float(
                                raw[name]
                            )
                            for name
                            in flow_features
                        ),
                )
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
                }
            )

    if len(rows) != 10840:
        raise RuntimeError(
            "Expected 10,840 OOS rows."
        )

    return rows


def make_price_tensors(
    training: list[Example],
    evaluation: list[Example],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    standardizer = (
        FeatureStandardizer.fit(
            [
                example.price_x
                for example
                in training
            ]
        )
    )

    train_x = torch.from_numpy(
        standardizer.transform(
            [
                example.price_x
                for example
                in training
            ]
        )
    )

    eval_x = torch.from_numpy(
        standardizer.transform(
            [
                example.price_x
                for example
                in evaluation
            ]
        )
    )

    target = torch.tensor(
        [
            example.target
            for example
            in training
        ],
        dtype=torch.float32,
    )

    return (
        train_x,
        target,
        eval_x,
    )


def make_multimodal_tensors(
    training: list[Example],
    evaluation: list[Example],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    price_scaler = (
        FeatureStandardizer.fit(
            [
                example.price_x
                for example
                in training
            ]
        )
    )

    flow_scaler = (
        FeatureStandardizer.fit(
            [
                example.flow_x
                for example
                in training
            ]
        )
    )

    price_train = torch.from_numpy(
        price_scaler.transform(
            [
                example.price_x
                for example
                in training
            ]
        )
    )

    flow_train = torch.from_numpy(
        flow_scaler.transform(
            [
                example.flow_x
                for example
                in training
            ]
        )
    )

    target = torch.tensor(
        [
            example.target
            for example
            in training
        ],
        dtype=torch.float32,
    )

    price_eval = torch.from_numpy(
        price_scaler.transform(
            [
                example.price_x
                for example
                in evaluation
            ]
        )
    )

    flow_eval = torch.from_numpy(
        flow_scaler.transform(
            [
                example.flow_x
                for example
                in evaluation
            ]
        )
    )

    return (
        price_train,
        flow_train,
        target,
        price_eval,
        flow_eval,
    )


def train_price_epoch(
    *,
    model: PriceLatentModel,
    optimizer: torch.optim.Optimizer,
    price_x: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    record_gradients: bool,
    fold: int,
    epoch: int,
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
            BATCH_SIZE,
        )
    ):
        indices = permutation[
            start:
            start + BATCH_SIZE
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            price_x[
                indices
            ]
        )

        loss = criterion(
            logits,
            target[
                indices
            ],
        )

        loss.backward()

        if record_gradients:
            rows.append(
                {
                    "model":
                        "PRICE_NN",
                    "fold":
                        fold,
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
                    "total_norm":
                        parameter_gradient_norm(
                            model.parameters()
                        ),
                    "price_encoder_norm":
                        parameter_gradient_norm(
                            model
                            .price_encoder
                            .parameters()
                        ),
                    "latent_adapter_norm":
                        parameter_gradient_norm(
                            model
                            .latent_adapter
                            .parameters()
                        ),
                    "flow_encoder_norm":
                        "",
                    "fusion_norm":
                        "",
                    "prediction_head_norm":
                        parameter_gradient_norm(
                            model
                            .prediction_head
                            .parameters()
                        ),
                }
            )

        optimizer.step()

    return rows


def train_multimodal_epoch(
    *,
    model: MultimodalLatentModel,
    optimizer: torch.optim.Optimizer,
    price_x: torch.Tensor,
    flow_x: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    record_gradients: bool,
    fold: int,
    epoch: int,
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
            BATCH_SIZE,
        )
    ):
        indices = permutation[
            start:
            start + BATCH_SIZE
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

        if record_gradients:
            rows.append(
                {
                    "model":
                        "MULTIMODAL_NN",
                    "fold":
                        fold,
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
                    "total_norm":
                        parameter_gradient_norm(
                            model.parameters()
                        ),
                    "price_encoder_norm":
                        parameter_gradient_norm(
                            model
                            .price_encoder
                            .parameters()
                        ),
                    "latent_adapter_norm":
                        "",
                    "flow_encoder_norm":
                        parameter_gradient_norm(
                            model
                            .flow_encoder
                            .parameters()
                        ),
                    "fusion_norm":
                        parameter_gradient_norm(
                            model
                            .fusion
                            .parameters()
                        ),
                    "prediction_head_norm":
                        parameter_gradient_norm(
                            model
                            .prediction_head
                            .parameters()
                        ),
                }
            )

        optimizer.step()

    return rows


def validation_log_loss(
    targets: list[int],
    probabilities: torch.Tensor,
) -> float:
    values = [
        float(value)
        for value
        in probabilities
        .cpu()
        .tolist()
    ]

    return mean(
        observation_log_loss(
            target,
            probability,
        )
        for target, probability
        in zip(
            targets,
            values,
            strict=True,
        )
    )


def select_price_epochs(
    inner_train: list[Example],
    validation: list[Example],
    *,
    seed: int,
    fold: int,
) -> tuple[
    int,
    list[dict[str, object]],
]:
    (
        train_x,
        target,
        validation_x,
    ) = make_price_tensors(
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

    model = PriceLatentModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 100000
    )

    best_epoch = 1

    best_loss = float(
        "inf"
    )

    curve = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        train_price_epoch(
            model=model,
            optimizer=optimizer,
            price_x=train_x,
            target=target,
            generator=generator,
            record_gradients=False,
            fold=fold,
            epoch=epoch,
        )

        model.eval()

        with torch.no_grad():
            probabilities = (
                model.predict_proba(
                    validation_x
                )
            )

        current_loss = (
            validation_log_loss(
                validation_targets,
                probabilities,
            )
        )

        curve.append(
            {
                "fold":
                    fold,
                "model":
                    "PRICE_NN",
                "epoch":
                    epoch,
                "validation_log_loss":
                    current_loss,
            }
        )

        if (
            current_loss
            < best_loss
            - 1e-12
        ):
            best_loss = (
                current_loss
            )

            best_epoch = (
                epoch
            )

    return (
        best_epoch,
        curve,
    )


def select_multimodal_epochs(
    inner_train: list[Example],
    validation: list[Example],
    *,
    seed: int,
    fold: int,
) -> tuple[
    int,
    list[dict[str, object]],
]:
    (
        price_train,
        flow_train,
        target,
        price_validation,
        flow_validation,
    ) = make_multimodal_tensors(
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 200000
    )

    best_epoch = 1

    best_loss = float(
        "inf"
    )

    curve = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        train_multimodal_epoch(
            model=model,
            optimizer=optimizer,
            price_x=price_train,
            flow_x=flow_train,
            target=target,
            generator=generator,
            record_gradients=False,
            fold=fold,
            epoch=epoch,
        )

        model.eval()

        with torch.no_grad():
            probabilities = (
                model.predict_proba(
                    price_validation,
                    flow_validation,
                )
            )

        current_loss = (
            validation_log_loss(
                validation_targets,
                probabilities,
            )
        )

        curve.append(
            {
                "fold":
                    fold,
                "model":
                    "MULTIMODAL_NN",
                "epoch":
                    epoch,
                "validation_log_loss":
                    current_loss,
            }
        )

        if (
            current_loss
            < best_loss
            - 1e-12
        ):
            best_loss = (
                current_loss
            )

            best_epoch = (
                epoch
            )

    return (
        best_epoch,
        curve,
    )


def refit_price(
    training: list[Example],
    test: list[Example],
    *,
    epochs: int,
    seed: int,
    fold: int,
) -> tuple[
    list[float],
    list[dict[str, object]],
]:
    (
        train_x,
        target,
        test_x,
    ) = make_price_tensors(
        training,
        test,
    )

    set_research_seed(
        seed
    )

    model = PriceLatentModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed + 300000
    )

    gradient_rows = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        gradient_rows.extend(
            train_price_epoch(
                model=model,
                optimizer=optimizer,
                price_x=train_x,
                target=target,
                generator=generator,
                record_gradients=True,
                fold=fold,
                epoch=epoch,
            )
        )

    model.eval()

    with torch.no_grad():
        probabilities = (
            model.predict_proba(
                test_x
            )
        )

    return (
        [
            float(value)
            for value
            in probabilities
            .cpu()
            .tolist()
        ],
        gradient_rows,
    )


def refit_multimodal(
    training: list[Example],
    test: list[Example],
    *,
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
        target,
        price_test,
        flow_test,
    ) = make_multimodal_tensors(
        training,
        test,
    )

    set_research_seed(
        seed
    )

    model = MultimodalLatentModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    generator = torch.Generator(
        device="cpu"
    )

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
            train_multimodal_epoch(
                model=model,
                optimizer=optimizer,
                price_x=price_train,
                flow_x=flow_train,
                target=target,
                generator=generator,
                record_gradients=True,
                fold=fold,
                epoch=epoch,
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
            float(value)
            for value
            in probabilities
            .cpu()
            .tolist()
        ],
        gradient_rows,
    )


def hac_standard_error(
    values: list[float],
    lag: int,
) -> float:
    n = len(values)

    mu = mean(values)

    centered = [
        value - mu
        for value
        in values
    ]

    variance = (
        sum(
            value * value
            for value
            in centered
        )
        / n
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

        variance += (
            2.0
            * weight
            * covariance
        )

    return math.sqrt(
        max(
            variance,
            0.0,
        )
        / n
    )


def normal_p(
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
            1.0 - cdf
        )
    )


def comparison(
    rows: list[dict[str, object]],
    first: str,
    second: str,
) -> dict[str, object]:
    log_diffs = []

    brier_diffs = []

    for row in rows:
        target = int(
            row[
                "target"
            ]
        )

        p_first = float(
            row[
                first
            ]
        )

        p_second = float(
            row[
                second
            ]
        )

        log_diffs.append(
            observation_log_loss(
                target,
                p_second,
            )
            - observation_log_loss(
                target,
                p_first,
            )
        )

        brier_diffs.append(
            observation_brier(
                target,
                p_second,
            )
            - observation_brier(
                target,
                p_first,
            )
        )

    result = {}

    for name, values in (
        (
            "log_loss",
            log_diffs,
        ),
        (
            "brier",
            brier_diffs,
        ),
    ):
        mu = mean(
            values
        )

        hac = {}

        for lag in HAC_LAGS:
            se = (
                hac_standard_error(
                    values,
                    lag,
                )
            )

            statistic = (
                mu / se
                if se > 0.0
                else 0.0
            )

            hac[
                str(lag)
            ] = {
                "standard_error":
                    se,
                "t_statistic":
                    statistic,
                "p_value":
                    normal_p(
                        statistic
                    ),
            }

        result[name] = {
            "mean_difference":
                mu,
            "hac":
                hac,
        }

    return result


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

    predictions = []

    selection_rows = []

    gradient_rows = []

    fold_rows = []

    print(
        "NESTED LATENT NEURAL WALK-FORWARD"
    )

    print(
        "================================="
    )

    for fold in range(20):
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

        if (
            len(outer_train)
            <= INNER_VALIDATION_OBSERVATIONS
        ):
            raise RuntimeError(
                "Outer training sample is "
                "too small for validation."
            )

        validation = (
            outer_train[
                -INNER_VALIDATION_OBSERVATIONS:
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
            for row in frozen
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

        seed = (
            BASE_SEED
            + fold
        )

        (
            price_epochs,
            price_curve,
        ) = select_price_epochs(
            inner_train,
            validation,
            seed=seed,
            fold=fold,
        )

        (
            multimodal_epochs,
            multimodal_curve,
        ) = (
            select_multimodal_epochs(
                inner_train,
                validation,
                seed=
                    seed + 50000,
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
        ) = refit_price(
            outer_train,
            outer_test,
            epochs=
                price_epochs,
            seed=
                seed + 1000000,
            fold=fold,
        )

        (
            multimodal_probabilities,
            multimodal_gradients,
        ) = refit_multimodal(
            outer_train,
            outer_test,
            epochs=
                multimodal_epochs,
            seed=
                seed + 2000000,
            fold=fold,
        )

        gradient_rows.extend(
            price_gradients
        )

        gradient_rows.extend(
            multimodal_gradients
        )

        fold_prediction_rows = []

        for (
            frozen_row,
            example,
            price_probability,
            multimodal_probability,
        ) in zip(
            frozen,
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
                "target":
                    example.target,
                "m0_probability":
                    frozen_row[
                        "m0_probability"
                    ],
                "linear_price_probability":
                    frozen_row[
                        "mprice_probability"
                    ],
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

        linear = metric_set(
            targets,
            [
                float(
                    row[
                        "linear_price_probability"
                    ]
                )
                for row
                in fold_prediction_rows
            ],
        )

        price_nn = metric_set(
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

        multimodal_nn = metric_set(
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

        fold_rows.append(
            {
                "fold":
                    fold,
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
                "test_observations":
                    len(
                        outer_test
                    ),
                "price_selected_epochs":
                    price_epochs,
                "multimodal_selected_epochs":
                    multimodal_epochs,
                "linear_price_log_loss":
                    linear[
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
                "price_nn_minus_linear":
                    (
                        price_nn[
                            "log_loss"
                        ]
                        - linear[
                            "log_loss"
                        ]
                    ),
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
            f"epochs Price="
            f"{price_epochs:2d} "
            f"Multi="
            f"{multimodal_epochs:2d}  "
            f"Linear="
            f"{linear['log_loss']:.6f} "
            f"PriceNN="
            f"{price_nn['log_loss']:.6f} "
            f"MultiNN="
            f"{multimodal_nn['log_loss']:.6f}"
        )

    targets = [
        int(
            row[
                "target"
            ]
        )
        for row
        in predictions
    ]

    aggregate = {
        "M0":
            metric_set(
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
        "M_PRICE_LINEAR":
            metric_set(
                targets,
                [
                    float(
                        row[
                            "linear_price_probability"
                        ]
                    )
                    for row
                    in predictions
                ],
            ),
        "M_PRICE_NN":
            metric_set(
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
            metric_set(
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

    price_vs_linear = comparison(
        predictions,
        "linear_price_probability",
        "price_nn_probability",
    )

    multimodal_vs_price = comparison(
        predictions,
        "price_nn_probability",
        "multimodal_nn_probability",
    )

    results = {
        "experiment":
            "nested-latent-neural-walk-forward-4h",
        "protocol": {
            "outer_folds":
                20,
            "outer_oos_forecasts":
                len(
                    predictions
                ),
            "inner_validation_observations":
                INNER_VALIDATION_OBSERVATIONS,
            "maximum_epochs":
                MAX_EPOCHS,
            "epoch_selection":
                (
                    "minimum chronological "
                    "inner-validation log loss"
                ),
            "refit":
                (
                    "from scratch on complete "
                    "legal outer training set"
                ),
            "point_in_time":
                True,
            "hyperparameter_search":
                False,
        },
        "metrics":
            aggregate,
        "price_nn_vs_linear_price":
            price_vs_linear,
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

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                predictions[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            predictions
        )

    with SELECTION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                selection_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            selection_rows
        )

    with GRADIENT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                gradient_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            gradient_rows
        )

    with FOLD_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fold_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            fold_rows
        )

    print()
    print(
        "AGGREGATE NESTED RESULTS"
    )

    print(
        "------------------------"
    )

    for name in (
        "M0",
        "M_PRICE_LINEAR",
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
        "PRICE NN - LINEAR PRICE"
    )

    print(
        "-----------------------"
    )

    item = price_vs_linear[
        "log_loss"
    ]

    print(
        f"LL delta: "
        f"{item['mean_difference']:+.8f}"
    )

    for lag in HAC_LAGS:
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
        "MULTIMODAL NN - PRICE NN"
    )

    print(
        "------------------------"
    )

    item = multimodal_vs_price[
        "log_loss"
    ]

    print(
        f"LL delta: "
        f"{item['mean_difference']:+.8f}"
    )

    for lag in HAC_LAGS:
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
        "Results ->",
        RESULTS_FILE,
    )


if __name__ == "__main__":
    main()
