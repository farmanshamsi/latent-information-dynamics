from __future__ import annotations

import csv
import json
import math

from dataclasses import dataclass
from datetime import datetime
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
    "latent_neural/first_fold"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "first_latent_neural_fold.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "first_latent_neural_fold_predictions.csv"
)

GRADIENT_FILE = (
    OUTPUT_ROOT
    / "first_latent_neural_fold_gradients.csv"
)

EPOCH_FILE = (
    OUTPUT_ROOT
    / "first_latent_neural_fold_epochs.csv"
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

EXPECTED_TRAIN_OBSERVATIONS = 2190

EXPECTED_TEST_OBSERVATIONS = 546

SEED = 20261016

EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

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


def probability_log_loss(
    targets: list[int],
    probabilities: list[float],
) -> float:
    losses = []

    for target, probability in zip(
        targets,
        probabilities,
        strict=True,
    ):
        p = min(
            max(
                probability,
                EPSILON,
            ),
            1.0 - EPSILON,
        )

        losses.append(
            -(
                target
                * math.log(
                    p
                )
                + (
                    1 - target
                )
                * math.log(
                    1.0 - p
                )
            )
        )

    return mean(
        losses
    )


def brier_score(
    targets: list[int],
    probabilities: list[float],
) -> float:
    return mean(
        (
            probability
            - target
        ) ** 2
        for target, probability
        in zip(
            targets,
            probabilities,
            strict=True,
        )
    )


def accuracy(
    targets: list[int],
    probabilities: list[float],
) -> float:
    return mean(
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
    )


def metric_set(
    targets: list[int],
    probabilities: list[float],
) -> dict[str, float]:
    return {
        "log_loss":
            probability_log_loss(
                targets,
                probabilities,
            ),
        "brier_score":
            brier_score(
                targets,
                probabilities,
            ),
        "accuracy":
            accuracy(
                targets,
                probabilities,
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
            "Expected exactly 31 features."
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
            "Combined dataset is missing "
            f"price features: {sorted(missing_price)}"
        )

    flow_features = tuple(
        name
        for name
        in feature_names
        if name
        not in PRICE_FEATURES
    )

    if len(
        flow_features
    ) != 20:
        raise RuntimeError(
            "Expected exactly 20 flow features, "
            f"got {len(flow_features)}."
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

        required = {
            "decision_time",
            "label_available_time",
            "target",
            *feature_names,
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
                "Supervised dataset missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:
            examples.append(
                Example(
                    decision_time=
                        parse_datetime(
                            row[
                                "decision_time"
                            ]
                        ),
                    label_available_time=
                        parse_datetime(
                            row[
                                "label_available_time"
                            ]
                        ),
                    target=
                        int(
                            row[
                                "target"
                            ]
                        ),
                    price_x=
                        tuple(
                            float(
                                row[name]
                            )
                            for name
                            in PRICE_FEATURES
                        ),
                    flow_x=
                        tuple(
                            float(
                                row[name]
                            )
                            for name
                            in flow_features
                        ),
                )
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
                "Examples are not strictly "
                "chronological."
            )

    return (
        examples,
        flow_features,
    )


def load_baseline_fold_zero() -> list[
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
                "Frozen baseline file missing "
                f"columns: {sorted(missing)}"
            )

        for raw in reader:
            if int(
                raw[
                    "fold"
                ]
            ) != 0:
                continue

            rows.append(
                {
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
    ) != EXPECTED_TEST_OBSERVATIONS:
        raise RuntimeError(
            "Frozen fold 0 contains "
            f"{len(rows)} observations, "
            f"expected {EXPECTED_TEST_OBSERVATIONS}."
        )

    return rows


def gradient_summary(
    gradient_rows: list[
        dict[str, float | int]
    ],
) -> dict[str, object]:
    total = [
        float(
            row[
                "total_norm"
            ]
        )
        for row
        in gradient_rows
    ]

    price = [
        float(
            row[
                "price_encoder_norm"
            ]
        )
        for row
        in gradient_rows
    ]

    flow = [
        float(
            row[
                "flow_encoder_norm"
            ]
        )
        for row
        in gradient_rows
    ]

    fusion = [
        float(
            row[
                "fusion_norm"
            ]
        )
        for row
        in gradient_rows
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

    return {
        "steps":
            len(
                gradient_rows
            ),
        "total_norm": {
            "mean":
                mean(
                    total
                ),
            "median":
                percentile(
                    total,
                    0.50,
                ),
            "p95":
                percentile(
                    total,
                    0.95,
                ),
            "p99":
                percentile(
                    total,
                    0.99,
                ),
            "maximum":
                max(
                    total
                ),
        },
        "price_encoder_norm": {
            "mean":
                mean(
                    price
                ),
            "p95":
                percentile(
                    price,
                    0.95,
                ),
            "maximum":
                max(
                    price
                ),
        },
        "flow_encoder_norm": {
            "mean":
                mean(
                    flow
                ),
            "p95":
                percentile(
                    flow,
                    0.95,
                ),
            "maximum":
                max(
                    flow
                ),
        },
        "fusion_norm": {
            "mean":
                mean(
                    fusion
                ),
            "p95":
                percentile(
                    fusion,
                    0.95,
                ),
            "maximum":
                max(
                    fusion
                ),
        },
        "flow_to_price_gradient_ratio": {
            "mean":
                mean(
                    ratios
                ),
            "median":
                percentile(
                    ratios,
                    0.50,
                ),
            "p95":
                percentile(
                    ratios,
                    0.95,
                ),
            "maximum":
                max(
                    ratios
                ),
        },
        "nonfinite_gradient_elements":
            sum(
                int(
                    row[
                        "nonfinite_gradient_elements"
                    ]
                )
                for row
                in gradient_rows
            ),
    }


def main() -> None:
    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )

    set_research_seed(
        SEED
    )

    (
        examples,
        flow_features,
    ) = load_examples()

    baseline_rows = (
        load_baseline_fold_zero()
    )

    test_start = min(
        row[
            "decision_time"
        ]
        for row
        in baseline_rows
    )

    test_times = {
        row[
            "decision_time"
        ]
        for row
        in baseline_rows
    }

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

    example_by_time = {
        example.decision_time:
            example
        for example
        in examples
    }

    test_examples = []

    for baseline in baseline_rows:
        decision_time = baseline[
            "decision_time"
        ]

        example = example_by_time[
            decision_time
        ]

        if (
            example.target
            != baseline[
                "target"
            ]
        ):
            raise RuntimeError(
                "Target mismatch with frozen "
                "baseline fold."
            )

        test_examples.append(
            example
        )

    if len(
        train_examples
    ) != EXPECTED_TRAIN_OBSERVATIONS:
        raise RuntimeError(
            "Legal fold-0 training set has "
            f"{len(train_examples)} observations; "
            f"expected {EXPECTED_TRAIN_OBSERVATIONS}."
        )

    if len(
        test_examples
    ) != EXPECTED_TEST_OBSERVATIONS:
        raise RuntimeError(
            "Fold-0 test set has "
            f"{len(test_examples)} observations."
        )

    if any(
        example.label_available_time
        > test_start
        for example
        in train_examples
    ):
        raise RuntimeError(
            "PIT legality failure in training set."
        )

    if (
        {
            example.decision_time
            for example
            in test_examples
        }
        != test_times
    ):
        raise RuntimeError(
            "Neural test timestamps differ "
            "from frozen baseline."
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

    price_train_np = (
        price_standardizer.transform(
            [
                example.price_x
                for example
                in train_examples
            ]
        )
    )

    flow_train_np = (
        flow_standardizer.transform(
            [
                example.flow_x
                for example
                in train_examples
            ]
        )
    )

    price_test_np = (
        price_standardizer.transform(
            [
                example.price_x
                for example
                in test_examples
            ]
        )
    )

    flow_test_np = (
        flow_standardizer.transform(
            [
                example.flow_x
                for example
                in test_examples
            ]
        )
    )

    target_train_np = np.asarray(
        [
            example.target
            for example
            in train_examples
        ],
        dtype=np.float32,
    )

    target_test = [
        example.target
        for example
        in test_examples
    ]

    price_train = torch.from_numpy(
        price_train_np
    )

    flow_train = torch.from_numpy(
        flow_train_np
    )

    target_train = torch.from_numpy(
        target_train_np
    )

    price_test = torch.from_numpy(
        price_test_np
    )

    flow_test = torch.from_numpy(
        flow_test_np
    )

    model = MultimodalLatentModel()

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        SEED
    )

    gradient_rows = []

    epoch_rows = []

    global_step = 0

    for epoch in range(
        EPOCHS
    ):
        permutation = torch.randperm(
            len(
                train_examples
            ),
            generator=generator,
        )

        epoch_weighted_loss = 0.0

        epoch_observations = 0

        epoch_gradient_norms = []

        for batch_number, start in enumerate(
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

            result = (
                binary_training_step(
                    model=
                        model,
                    optimizer=
                        optimizer,
                    price_x=
                        batch_price,
                    flow_x=
                        batch_flow,
                    target=
                        batch_target,
                )
            )

            batch_size = int(
                indices.shape[0]
            )

            epoch_weighted_loss += (
                result.loss
                * batch_size
            )

            epoch_observations += (
                batch_size
            )

            snapshot = (
                result.gradients
            )

            epoch_gradient_norms.append(
                snapshot.total_norm
            )

            gradient_rows.append(
                {
                    "global_step":
                        global_step,
                    "epoch":
                        epoch,
                    "batch":
                        batch_number,
                    "batch_observations":
                        batch_size,
                    "loss":
                        result.loss,
                    **snapshot.as_dict(),
                }
            )

            global_step += 1

        epoch_rows.append(
            {
                "epoch":
                    epoch,
                "mean_training_loss":
                    (
                        epoch_weighted_loss
                        / epoch_observations
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
        )

    model.eval()

    with torch.no_grad():
        output = (
            model.forward_with_state(
                price_test,
                flow_test,
            )
        )

        probabilities_tensor = (
            torch.sigmoid(
                output.logits
            )
        )

    probabilities = [
        float(value)
        for value
        in probabilities_tensor.tolist()
    ]

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

    baseline_m0 = [
        float(
            row[
                "m0_probability"
            ]
        )
        for row
        in baseline_rows
    ]

    baseline_price = [
        float(
            row[
                "mprice_probability"
            ]
        )
        for row
        in baseline_rows
    ]

    baseline_combined = [
        float(
            row[
                "mcombined_probability"
            ]
        )
        for row
        in baseline_rows
    ]

    prediction_rows = []

    for index, (
        example,
        probability,
    ) in enumerate(
        zip(
            test_examples,
            probabilities,
            strict=True,
        )
    ):
        row = {
            "fold":
                0,
            "decision_time":
                example
                .decision_time
                .isoformat(),
            "target":
                example.target,
            "m0_probability":
                baseline_m0[
                    index
                ],
            "mprice_probability":
                baseline_price[
                    index
                ],
            "mcombined_probability":
                baseline_combined[
                    index
                ],
            "mlatent_probability":
                probability,
        }

        for state_index in range(
            price_states.shape[1]
        ):
            row[
                f"price_state_{state_index}"
            ] = float(
                price_states[
                    index,
                    state_index,
                ]
            )

        for state_index in range(
            flow_states.shape[1]
        ):
            row[
                f"flow_state_{state_index}"
            ] = float(
                flow_states[
                    index,
                    state_index,
                ]
            )

        for state_index in range(
            latent_states.shape[1]
        ):
            row[
                f"latent_state_{state_index}"
            ] = float(
                latent_states[
                    index,
                    state_index,
                ]
            )

        prediction_rows.append(
            row
        )

    metrics = {
        "M0":
            metric_set(
                target_test,
                baseline_m0,
            ),
        "M_PRICE":
            metric_set(
                target_test,
                baseline_price,
            ),
        "M_COMBINED_RAW":
            metric_set(
                target_test,
                baseline_combined,
            ),
        "M_LATENT":
            metric_set(
                target_test,
                probabilities,
            ),
    }

    results = {
        "experiment":
            "first-latent-neural-fold",
        "purpose":
            (
                "Architecture, PIT, scaling, "
                "training and gradient-instrumentation "
                "validation. Not a final OOS result."
            ),
        "fold":
            0,
        "device":
            "cpu",
        "seed":
            SEED,
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
        "point_in_time": {
            "rule":
                (
                    "decision_time < test_start and "
                    "label_available_time <= test_start"
                ),
            "passed":
                True,
        },
        "standardization": {
            "fit_scope":
                "outer fold training observations only",
            "price":
                price_standardizer.as_dict(),
            "flow":
                flow_standardizer.as_dict(),
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
        "optimizer": {
            "name":
                "AdamW",
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "epochs":
                EPOCHS,
            "batch_size":
                BATCH_SIZE,
            "hyperparameter_search":
                False,
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
            metrics,
        "gradient_diagnostics":
            gradient_summary(
                gradient_rows
            ),
        "training": {
            "first_epoch":
                epoch_rows[
                    0
                ],
            "last_epoch":
                epoch_rows[
                    -1
                ],
        },
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
                prediction_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            prediction_rows
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

    with EPOCH_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                epoch_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            epoch_rows
        )

    print(
        "FIRST LATENT NEURAL FOLD"
    )

    print(
        "------------------------"
    )

    print(
        f"Test start: "
        f"{test_start.isoformat()}"
    )

    print(
        f"Train observations: "
        f"{len(train_examples):,}"
    )

    print(
        f"Test observations: "
        f"{len(test_examples):,}"
    )

    print(
        "PIT audit: PASS"
    )

    print()

    for name in (
        "M0",
        "M_PRICE",
        "M_COMBINED_RAW",
        "M_LATENT",
    ):
        item = metrics[
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

    gradients = results[
        "gradient_diagnostics"
    ]

    print()
    print(
        "GRADIENT DIAGNOSTICS"
    )
    print(
        "--------------------"
    )

    print(
        "Steps: "
        f"{gradients['steps']:,}"
    )

    print(
        "Total norm mean: "
        f"{gradients['total_norm']['mean']:.6f}"
    )

    print(
        "Total norm p95:  "
        f"{gradients['total_norm']['p95']:.6f}"
    )

    print(
        "Total norm max:  "
        f"{gradients['total_norm']['maximum']:.6f}"
    )

    print(
        "Price grad mean: "
        f"{gradients['price_encoder_norm']['mean']:.6f}"
    )

    print(
        "Flow grad mean:  "
        f"{gradients['flow_encoder_norm']['mean']:.6f}"
    )

    print(
        "Flow/Price mean: "
        f"{gradients['flow_to_price_gradient_ratio']['mean']:.6f}"
    )

    print(
        "Non-finite:      "
        f"{gradients['nonfinite_gradient_elements']}"
    )

    print()
    print(
        "TRAINING LOSS"
    )
    print(
        "-------------"
    )

    print(
        "Epoch 0:  "
        f"{epoch_rows[0]['mean_training_loss']:.8f}"
    )

    print(
        "Epoch 49: "
        f"{epoch_rows[-1]['mean_training_loss']:.8f}"
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


if __name__ == "__main__":
    main()
