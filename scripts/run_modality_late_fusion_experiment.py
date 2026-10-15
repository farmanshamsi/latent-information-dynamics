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

from latent_information_dynamics.models.binary import (
    LogisticRegressionBinary,
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

WARM_FILE = Path(
    "data/derived/experiments/"
    "probability_shrinkage/"
    "warm_start_probability_shrinkage_predictions.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "modality_late_fusion"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "modality_late_fusion_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "modality_late_fusion_predictions.csv"
)

SCORE_STREAM_FILE = (
    OUTPUT_ROOT
    / "modality_score_stream.csv"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "modality_late_fusion_folds.csv"
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

BASE_MODEL_L2 = 1.0
META_MODEL_L2 = 1.0

# Same historical score-generation protocol
# used by the warm-start calibration stage.
SCORE_MIN_TRAIN = 1095
SCORE_TEST_OBSERVATIONS = 273

HAC_LAGS = (
    0,
    1,
    6,
    12,
    42,
)

EPSILON = 1e-12

VALIDATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
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


def clip_probability(
    probability: float,
) -> float:
    return min(
        max(
            probability,
            EPSILON,
        ),
        1.0 - EPSILON,
    )


def logit(
    probability: float,
) -> float:
    p = clip_probability(
        probability
    )

    return math.log(
        p / (1.0 - p)
    )


def observation_log_loss(
    target: int,
    probability: float,
) -> float:
    p = clip_probability(
        probability
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


def price_meta_features(
    p0: float,
    price_probability: float,
) -> tuple[float, ...]:
    base_logit = logit(
        p0
    )

    return (
        base_logit,
        logit(
            price_probability
        )
        - base_logit,
    )


def fusion_meta_features(
    p0: float,
    price_probability: float,
    flow_probability: float,
) -> tuple[float, ...]:
    base_logit = logit(
        p0
    )

    return (
        base_logit,
        logit(
            price_probability
        )
        - base_logit,
        logit(
            flow_probability
        )
        - base_logit,
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
            "Expected exactly 20 "
            "trade-flow features, got "
            f"{len(flow_features)}."
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

    if not examples:
        raise RuntimeError(
            "No examples loaded."
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
                "Dataset is not strictly "
                "chronological."
            )

    return (
        examples,
        flow_features,
    )


def fit_modality(
    train_examples: list[Example],
    test_examples: list[Example],
    *,
    modality: str,
) -> tuple[float, ...]:
    if modality == "price":
        train_X = tuple(
            example.price_x
            for example
            in train_examples
        )

        test_X = tuple(
            example.price_x
            for example
            in test_examples
        )

    elif modality == "flow":
        train_X = tuple(
            example.flow_x
            for example
            in train_examples
        )

        test_X = tuple(
            example.flow_x
            for example
            in test_examples
        )

    else:
        raise ValueError(
            f"Unknown modality: {modality}"
        )

    train_y = tuple(
        example.target
        for example
        in train_examples
    )

    model = LogisticRegressionBinary(
        l2=BASE_MODEL_L2,
    )

    model.fit(
        train_X,
        train_y,
    )

    return tuple(
        model.predict_proba(
            test_X
        )
    )


def build_score_stream(
    examples: list[Example],
) -> list[dict[str, object]]:
    records = []

    candidate = (
        SCORE_MIN_TRAIN
    )

    score_fold = 0

    while candidate < len(
        examples
    ):
        test_start = (
            examples[
                candidate
            ].decision_time
        )

        train_indices = tuple(
            index
            for index, example
            in enumerate(
                examples[
                    :candidate
                ]
            )
            if (
                example.label_available_time
                <= test_start
            )
        )

        if len(
            train_indices
        ) < SCORE_MIN_TRAIN:
            candidate += 1
            continue

        test_end = min(
            candidate
            + SCORE_TEST_OBSERVATIONS,
            len(
                examples
            ),
        )

        test_indices = tuple(
            range(
                candidate,
                test_end,
            )
        )

        train_examples = [
            examples[
                index
            ]
            for index
            in train_indices
        ]

        test_examples = [
            examples[
                index
            ]
            for index
            in test_indices
        ]

        p0 = mean(
            example.target
            for example
            in train_examples
        )

        price_probabilities = (
            fit_modality(
                train_examples,
                test_examples,
                modality="price",
            )
        )

        flow_probabilities = (
            fit_modality(
                train_examples,
                test_examples,
                modality="flow",
            )
        )

        for (
            example,
            price_probability,
            flow_probability,
        ) in zip(
            test_examples,
            price_probabilities,
            flow_probabilities,
            strict=True,
        ):
            records.append(
                {
                    "score_fold":
                        score_fold,
                    "decision_time":
                        example.decision_time,
                    "label_available_time":
                        example.label_available_time,
                    "target":
                        example.target,
                    "m0_probability":
                        p0,
                    "price_probability":
                        float(
                            price_probability
                        ),
                    "flow_probability":
                        float(
                            flow_probability
                        ),
                }
            )

        score_fold += 1

        candidate = test_end

    if not records:
        raise RuntimeError(
            "No modality score forecasts "
            "were produced."
        )

    return records


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
            "minfo_probability",
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
                "Baseline file missing "
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
                    "minfo_probability":
                        float(
                            raw[
                                "minfo_probability"
                            ]
                        ),
                }
            )

    if len(
        rows
    ) != 10840:
        raise RuntimeError(
            "Expected 10,840 baseline "
            f"rows, got {len(rows)}."
        )

    return rows


def load_warm_price() -> dict[
    tuple[int, datetime],
    float,
]:
    output = {}

    with WARM_FILE.open(
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
            "mprice_warm_probability",
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
                "Warm-start file missing "
                f"columns: {sorted(missing)}"
            )

        for raw in reader:
            key = (
                int(
                    raw[
                        "fold"
                    ]
                ),
                parse_datetime(
                    raw[
                        "decision_time"
                    ]
                ),
            )

            output[
                key
            ] = float(
                raw[
                    "mprice_warm_probability"
                ]
            )

    return output


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
        for row in rows
    ]

    targets = [
        int(
            row[
                "target"
            ]
        )
        for row in rows
    ]

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
                observation_accuracy(
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
        "mean_probability":
            mean(
                probabilities
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

    gamma0 = (
        sum(
            value * value
            for value
            in centered
        )
        / n
    )

    long_run_variance = (
        gamma0
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
    logloss_difference = []

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

        logloss_difference.append(
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

    output = {}

    for name, values in (
        (
            "log_loss",
            logloss_difference,
        ),
        (
            "brier",
            brier_difference,
        ),
    ):
        difference_mean = mean(
            values
        )

        lag_results = {}

        for lag in HAC_LAGS:
            se = hac_standard_error(
                values,
                lag,
            )

            statistic = (
                difference_mean / se
                if se > 0.0
                else 0.0
            )

            lag_results[
                str(
                    lag
                )
            ] = {
                "standard_error":
                    se,
                "t_statistic":
                    statistic,
                "p_value":
                    two_sided_normal_p(
                        statistic
                    ),
            }

        output[
            name
        ] = {
            "mean_difference":
                difference_mean,
            "hac":
                lag_results,
        }

    return output


def main() -> None:
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

    print(
        "Building historical "
        "modality-score stream..."
    )

    score_stream = (
        build_score_stream(
            examples
        )
    )

    baseline_rows = (
        load_baseline_rows()
    )

    warm_price = (
        load_warm_price()
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

    if len(
        folds
    ) != 20:
        raise RuntimeError(
            f"Expected 20 folds, "
            f"got {len(folds)}."
        )

    output_rows = []

    fold_rows_output = []

    max_price_reproduction_error = 0.0
    max_flow_reproduction_error = 0.0

    for fold in folds:
        fold_rows = grouped[
            fold
        ]

        test_start = min(
            row[
                "decision_time"
            ]
            for row
            in fold_rows
        )

        outer_train = [
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

        if not outer_train:
            raise RuntimeError(
                f"No outer training data "
                f"for fold {fold}."
            )

        test_examples = []

        for row in fold_rows:
            decision_time = row[
                "decision_time"
            ]

            if (
                decision_time
                not in example_by_time
            ):
                raise RuntimeError(
                    "Missing outer test "
                    f"example {decision_time}."
                )

            example = (
                example_by_time[
                    decision_time
                ]
            )

            if (
                example.target
                != int(
                    row[
                        "target"
                    ]
                )
            ):
                raise RuntimeError(
                    "Target mismatch."
                )

            test_examples.append(
                example
            )

        p0 = mean(
            example.target
            for example
            in outer_train
        )

        expected_p0 = float(
            fold_rows[
                0
            ][
                "m0_probability"
            ]
        )

        if not math.isclose(
            p0,
            expected_p0,
            rel_tol=0.0,
            abs_tol=VALIDATION_TOLERANCE,
        ):
            raise RuntimeError(
                f"Fold {fold}: reconstructed "
                "M0 differs from frozen M0. "
                f"{p0} vs {expected_p0}"
            )

        price_probabilities = (
            fit_modality(
                outer_train,
                test_examples,
                modality="price",
            )
        )

        flow_probabilities = (
            fit_modality(
                outer_train,
                test_examples,
                modality="flow",
            )
        )

        for (
            row,
            price_probability,
            flow_probability,
        ) in zip(
            fold_rows,
            price_probabilities,
            flow_probabilities,
            strict=True,
        ):
            price_error = abs(
                float(
                    price_probability
                )
                - float(
                    row[
                        "mprice_probability"
                    ]
                )
            )

            flow_error = abs(
                float(
                    flow_probability
                )
                - float(
                    row[
                        "minfo_probability"
                    ]
                )
            )

            max_price_reproduction_error = max(
                max_price_reproduction_error,
                price_error,
            )

            max_flow_reproduction_error = max(
                max_flow_reproduction_error,
                flow_error,
            )

            if (
                price_error
                > VALIDATION_TOLERANCE
            ):
                raise RuntimeError(
                    f"Fold {fold}: price "
                    "encoder does not reproduce "
                    "frozen baseline."
                )

            if (
                flow_error
                > VALIDATION_TOLERANCE
            ):
                raise RuntimeError(
                    f"Fold {fold}: flow "
                    "encoder does not reproduce "
                    "frozen baseline."
                )

        fusion_history = [
            record
            for record
            in score_stream
            if (
                record[
                    "label_available_time"
                ]
                <= test_start
            )
        ]

        if not fusion_history:
            raise RuntimeError(
                f"No legal fusion history "
                f"for fold {fold}."
            )

        price_meta_X = tuple(
            price_meta_features(
                float(
                    record[
                        "m0_probability"
                    ]
                ),
                float(
                    record[
                        "price_probability"
                    ]
                ),
            )
            for record
            in fusion_history
        )

        fusion_meta_X = tuple(
            fusion_meta_features(
                float(
                    record[
                        "m0_probability"
                    ]
                ),
                float(
                    record[
                        "price_probability"
                    ]
                ),
                float(
                    record[
                        "flow_probability"
                    ]
                ),
            )
            for record
            in fusion_history
        )

        meta_y = tuple(
            int(
                record[
                    "target"
                ]
            )
            for record
            in fusion_history
        )

        price_meta_model = (
            LogisticRegressionBinary(
                l2=META_MODEL_L2,
            )
        )

        fusion_model = (
            LogisticRegressionBinary(
                l2=META_MODEL_L2,
            )
        )

        price_meta_model.fit(
            price_meta_X,
            meta_y,
        )

        fusion_model.fit(
            fusion_meta_X,
            meta_y,
        )

        price_meta_test_X = tuple(
            price_meta_features(
                p0,
                float(
                    price_probability
                ),
            )
            for price_probability
            in price_probabilities
        )

        fusion_meta_test_X = tuple(
            fusion_meta_features(
                p0,
                float(
                    price_probability
                ),
                float(
                    flow_probability
                ),
            )
            for (
                price_probability,
                flow_probability,
            )
            in zip(
                price_probabilities,
                flow_probabilities,
                strict=True,
            )
        )

        price_meta_probabilities = (
            price_meta_model.predict_proba(
                price_meta_test_X
            )
        )

        fusion_probabilities = (
            fusion_model.predict_proba(
                fusion_meta_test_X
            )
        )

        current_output_rows = []

        for (
            row,
            price_probability,
            flow_probability,
            price_meta_probability,
            fusion_probability,
        ) in zip(
            fold_rows,
            price_probabilities,
            flow_probabilities,
            price_meta_probabilities,
            fusion_probabilities,
            strict=True,
        ):
            key = (
                fold,
                row[
                    "decision_time"
                ],
            )

            if key not in warm_price:
                raise RuntimeError(
                    "Missing warm-price "
                    f"benchmark for {key}."
                )

            output = {
                "fold":
                    fold,
                "decision_time":
                    row[
                        "decision_time"
                    ].isoformat(),
                "target":
                    row[
                        "target"
                    ],
                "m0_probability":
                    p0,
                "mprice_probability":
                    float(
                        price_probability
                    ),
                "minfo_probability":
                    float(
                        flow_probability
                    ),
                "mprice_warm_probability":
                    warm_price[
                        key
                    ],
                "mprice_meta_probability":
                    float(
                        price_meta_probability
                    ),
                "mlate_fusion_probability":
                    float(
                        fusion_probability
                    ),
                "fusion_history_observations":
                    len(
                        fusion_history
                    ),
            }

            output_rows.append(
                output
            )

            current_output_rows.append(
                output
            )

        price_meta_metrics = metrics(
            current_output_rows,
            "mprice_meta_probability",
        )

        fusion_metrics = metrics(
            current_output_rows,
            "mlate_fusion_probability",
        )

        fold_rows_output.append(
            {
                "fold":
                    fold,
                "test_start":
                    test_start.isoformat(),
                "train_observations":
                    len(
                        outer_train
                    ),
                "fusion_history_observations":
                    len(
                        fusion_history
                    ),
                "price_meta_log_loss":
                    price_meta_metrics[
                        "log_loss"
                    ],
                "late_fusion_log_loss":
                    fusion_metrics[
                        "log_loss"
                    ],
                "log_loss_delta":
                    (
                        fusion_metrics[
                            "log_loss"
                        ]
                        - price_meta_metrics[
                            "log_loss"
                        ]
                    ),
                "price_meta_brier":
                    price_meta_metrics[
                        "brier_score"
                    ],
                "late_fusion_brier":
                    fusion_metrics[
                        "brier_score"
                    ],
                "brier_delta":
                    (
                        fusion_metrics[
                            "brier_score"
                        ]
                        - price_meta_metrics[
                            "brier_score"
                        ]
                    ),
                "price_meta_accuracy":
                    price_meta_metrics[
                        "accuracy"
                    ],
                "late_fusion_accuracy":
                    fusion_metrics[
                        "accuracy"
                    ],
                "accuracy_delta":
                    (
                        fusion_metrics[
                            "accuracy"
                        ]
                        - price_meta_metrics[
                            "accuracy"
                        ]
                    ),
                "price_meta_coefficients":
                    json.dumps(
                        list(
                            price_meta_model.coefficients_
                            or ()
                        )
                    ),
                "fusion_coefficients":
                    json.dumps(
                        list(
                            fusion_model.coefficients_
                            or ()
                        )
                    ),
            }
        )

    all_metrics = {
        "M0":
            metrics(
                output_rows,
                "m0_probability",
            ),
        "M_PRICE_RAW":
            metrics(
                output_rows,
                "mprice_probability",
            ),
        "M_INFO_RAW":
            metrics(
                output_rows,
                "minfo_probability",
            ),
        "M_PRICE_WARM":
            metrics(
                output_rows,
                "mprice_warm_probability",
            ),
        "M_PRICE_META":
            metrics(
                output_rows,
                "mprice_meta_probability",
            ),
        "M_LATE_FUSION":
            metrics(
                output_rows,
                "mlate_fusion_probability",
            ),
    }

    primary_significance = (
        hac_comparison(
            output_rows,
            "mprice_meta_probability",
            "mlate_fusion_probability",
        )
    )

    secondary_significance = (
        hac_comparison(
            output_rows,
            "mprice_warm_probability",
            "mlate_fusion_probability",
        )
    )

    fold_wins = {
        "log_loss":
            sum(
                row[
                    "log_loss_delta"
                ] < 0.0
                for row
                in fold_rows_output
            ),
        "brier":
            sum(
                row[
                    "brier_delta"
                ] < 0.0
                for row
                in fold_rows_output
            ),
        "accuracy":
            sum(
                row[
                    "accuracy_delta"
                ] > 0.0
                for row
                in fold_rows_output
            ),
    }

    results = {
        "experiment":
            "modality-specific-late-fusion-4h",
        "representation": {
            "price_state":
                (
                    "z_price = logit(p_price) "
                    "- logit(p0)"
                ),
            "flow_state":
                (
                    "z_flow = logit(p_flow) "
                    "- logit(p0)"
                ),
            "price_meta_features":
                [
                    "logit_p0",
                    "z_price",
                ],
            "late_fusion_features":
                [
                    "logit_p0",
                    "z_price",
                    "z_flow",
                ],
        },
        "protocol": {
            "base_model_l2":
                BASE_MODEL_L2,
            "meta_model_l2":
                META_MODEL_L2,
            "score_stream_min_train":
                SCORE_MIN_TRAIN,
            "score_stream_test_observations":
                SCORE_TEST_OBSERVATIONS,
            "hyperparameter_search":
                False,
            "point_in_time":
                True,
        },
        "price_feature_count":
            len(
                PRICE_FEATURES
            ),
        "flow_feature_count":
            len(
                flow_features
            ),
        "score_stream_forecasts":
            len(
                score_stream
            ),
        "outer_oos_forecasts":
            len(
                output_rows
            ),
        "baseline_reproduction": {
            "max_price_probability_error":
                max_price_reproduction_error,
            "max_flow_probability_error":
                max_flow_reproduction_error,
        },
        "metrics":
            all_metrics,
        "primary_comparison": {
            "first":
                "M_PRICE_META",
            "second":
                "M_LATE_FUSION",
            "difference_definition":
                "second_minus_first",
            "log_loss_delta":
                (
                    all_metrics[
                        "M_LATE_FUSION"
                    ][
                        "log_loss"
                    ]
                    - all_metrics[
                        "M_PRICE_META"
                    ][
                        "log_loss"
                    ]
                ),
            "brier_delta":
                (
                    all_metrics[
                        "M_LATE_FUSION"
                    ][
                        "brier_score"
                    ]
                    - all_metrics[
                        "M_PRICE_META"
                    ][
                        "brier_score"
                    ]
                ),
            "accuracy_delta":
                (
                    all_metrics[
                        "M_LATE_FUSION"
                    ][
                        "accuracy"
                    ]
                    - all_metrics[
                        "M_PRICE_META"
                    ][
                        "accuracy"
                    ]
                ),
            "combined_fold_wins":
                fold_wins,
            "hac":
                primary_significance,
        },
        "secondary_comparison_vs_warm_price":
            {
                "log_loss_delta":
                    (
                        all_metrics[
                            "M_LATE_FUSION"
                        ][
                            "log_loss"
                        ]
                        - all_metrics[
                            "M_PRICE_WARM"
                        ][
                            "log_loss"
                        ]
                    ),
                "brier_delta":
                    (
                        all_metrics[
                            "M_LATE_FUSION"
                        ][
                            "brier_score"
                        ]
                        - all_metrics[
                            "M_PRICE_WARM"
                        ][
                            "brier_score"
                        ]
                    ),
                "accuracy_delta":
                    (
                        all_metrics[
                            "M_LATE_FUSION"
                        ][
                            "accuracy"
                        ]
                        - all_metrics[
                            "M_PRICE_WARM"
                        ][
                            "accuracy"
                        ]
                    ),
                "hac":
                    secondary_significance,
            },
        "folds":
            fold_rows_output,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    with SCORE_STREAM_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "score_fold",
            "decision_time",
            "label_available_time",
            "target",
            "m0_probability",
            "price_probability",
            "flow_probability",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in score_stream:
            writer.writerow(
                {
                    **record,
                    "decision_time":
                        record[
                            "decision_time"
                        ].isoformat(),
                    "label_available_time":
                        record[
                            "label_available_time"
                        ].isoformat(),
                }
            )

    with FOLD_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fold_rows_output[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            fold_rows_output
        )

    print()
    print(
        "MODALITY-SPECIFIC LATE FUSION"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Score-stream forecasts: "
        f"{len(score_stream):,}"
    )

    print(
        f"Outer OOS forecasts: "
        f"{len(output_rows):,}"
    )

    print(
        "Baseline reproduction:"
    )

    print(
        "  max price error = "
        f"{max_price_reproduction_error:.3e}"
    )

    print(
        "  max flow error  = "
        f"{max_flow_reproduction_error:.3e}"
    )

    print()

    for name in (
        "M0",
        "M_PRICE_RAW",
        "M_INFO_RAW",
        "M_PRICE_WARM",
        "M_PRICE_META",
        "M_LATE_FUSION",
    ):
        item = all_metrics[
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
        "PRIMARY: LATE FUSION - PRICE META"
    )
    print(
        "---------------------------------"
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
        "Fusion fold wins: "
        f"LL {fold_wins['log_loss']}/20, "
        f"Brier {fold_wins['brier']}/20, "
        f"Acc {fold_wins['accuracy']}/20"
    )

    print()

    for metric in (
        "log_loss",
        "brier",
    ):
        item = primary_significance[
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

    secondary = results[
        "secondary_comparison_vs_warm_price"
    ]

    print()
    print(
        "SECONDARY: LATE FUSION - WARM PRICE"
    )
    print(
        "-----------------------------------"
    )

    print(
        "Log loss: "
        f"{secondary['log_loss_delta']:+.8f}"
    )

    print(
        "Brier:    "
        f"{secondary['brier_delta']:+.8f}"
    )

    print(
        "Accuracy: "
        f"{secondary['accuracy_delta']:+.4%}"
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
        "Score stream ->",
        SCORE_STREAM_FILE,
    )

    print(
        "Folds       ->",
        FOLD_FILE,
    )


if __name__ == "__main__":
    main()
