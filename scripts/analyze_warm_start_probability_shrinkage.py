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

BASELINE_PREDICTIONS_FILE = Path(
    "data/derived/experiments/"
    "baseline_information_decomposition/"
    "baseline_model_predictions_4h_2020_2025.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "probability_shrinkage"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "warm_start_probability_shrinkage_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "warm_start_probability_shrinkage_predictions.csv"
)

CALIBRATION_FILE = (
    OUTPUT_ROOT
    / "warm_start_calibration_stream.csv"
)

ALPHA_FILE = (
    OUTPUT_ROOT
    / "warm_start_alpha_schedule.csv"
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

LOGISTIC_L2 = 1.0

# Fixed from the frozen outer protocol:
# 1095 = 2190 / 2
# 273  = 546 / 2
CAL_MIN_TRAIN = 1095
CAL_TEST_OBSERVATIONS = 273

ALPHA_GRID = tuple(
    step / 100.0
    for step in range(101)
)

HAC_LAGS = (
    0,
    1,
    6,
    12,
    42,
)

EPSILON = 1e-15


@dataclass(frozen=True)
class Example:
    decision_time: datetime
    label_available_time: datetime
    target: int
    price_x: tuple[float, ...]
    combined_x: tuple[float, ...]


def parse_datetime(
    value: str,
) -> datetime:
    return datetime.fromisoformat(
        value
    )


def clip_probability(
    value: float,
) -> float:
    return min(
        max(
            value,
            EPSILON,
        ),
        1.0 - EPSILON,
    )


def shrink_probability(
    p0: float,
    model_probability: float,
    alpha: float,
) -> float:
    return clip_probability(
        p0
        + alpha
        * (
            model_probability
            - p0
        )
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
        + (
            1 - target
        )
        * math.log(
            1.0 - p
        )
    )


def observation_brier(
    target: int,
    probability: float,
) -> float:
    return (
        probability
        - target
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
            "Expected exactly 31 combined features."
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
            "Combined dataset missing price "
            f"features: {sorted(missing_price)}"
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
                "Combined supervised file "
                "missing columns: "
                f"{sorted(missing)}"
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
                    combined_x=
                        tuple(
                            float(
                                raw[name]
                            )
                            for name
                            in feature_names
                        ),
                )
            )

    if not examples:
        raise RuntimeError(
            "No supervised examples loaded."
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
        feature_names,
    )


def calibration_folds(
    examples: list[Example],
) -> list[
    tuple[
        tuple[int, ...],
        tuple[int, ...],
    ]
]:
    folds = []

    candidate = (
        CAL_MIN_TRAIN
    )

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
        ) < CAL_MIN_TRAIN:
            candidate += 1
            continue

        test_end = min(
            candidate
            + CAL_TEST_OBSERVATIONS,
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

        folds.append(
            (
                train_indices,
                test_indices,
            )
        )

        candidate = test_end

    if not folds:
        raise RuntimeError(
            "No calibration folds generated."
        )

    return folds


def fit_logistic(
    train_examples: list[Example],
    test_examples: list[Example],
    *,
    combined: bool,
) -> tuple[float, ...]:
    if combined:
        train_X = tuple(
            example.combined_x
            for example
            in train_examples
        )

        test_X = tuple(
            example.combined_x
            for example
            in test_examples
        )

    else:
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

    train_y = tuple(
        example.target
        for example
        in train_examples
    )

    model = LogisticRegressionBinary(
        l2=LOGISTIC_L2,
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


def build_calibration_stream(
    examples: list[Example],
) -> list[dict[str, object]]:
    folds = calibration_folds(
        examples
    )

    records = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        folds
    ):
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
            fit_logistic(
                train_examples,
                test_examples,
                combined=False,
            )
        )

        combined_probabilities = (
            fit_logistic(
                train_examples,
                test_examples,
                combined=True,
            )
        )

        for (
            example,
            price_probability,
            combined_probability,
        ) in zip(
            test_examples,
            price_probabilities,
            combined_probabilities,
            strict=True,
        ):
            records.append(
                {
                    "calibration_fold":
                        fold_number,
                    "decision_time":
                        example.decision_time,
                    "label_available_time":
                        example.label_available_time,
                    "target":
                        example.target,
                    "m0_probability":
                        p0,
                    "mprice_probability":
                        float(
                            price_probability
                        ),
                    "mcombined_probability":
                        float(
                            combined_probability
                        ),
                }
            )

    return records


def load_outer_predictions() -> list[
    dict[str, object]
]:
    rows = []

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
                "Baseline prediction file "
                "missing columns: "
                f"{sorted(missing)}"
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
            "Expected 10,840 outer "
            f"predictions, got {len(rows)}."
        )

    return rows


def choose_alpha(
    history: list[
        dict[str, object]
    ],
    model_column: str,
) -> tuple[
    float,
    float,
]:
    if not history:
        raise RuntimeError(
            "Warm-start calibration history "
            "is unexpectedly empty."
        )

    candidates = []

    for alpha in ALPHA_GRID:
        losses = []

        for row in history:
            probability = (
                shrink_probability(
                    float(
                        row[
                            "m0_probability"
                        ]
                    ),
                    float(
                        row[
                            model_column
                        ]
                    ),
                    alpha,
                )
            )

            losses.append(
                observation_log_loss(
                    int(
                        row[
                            "target"
                        ]
                    ),
                    probability,
                )
            )

        score = mean(
            losses
        )

        # Higher alpha is preferred
        # only on exact score ties.
        candidates.append(
            (
                score,
                -alpha,
                alpha,
            )
        )

    score, _, alpha = min(
        candidates
    )

    return (
        alpha,
        score,
    )


def metrics(
    rows: list[
        dict[str, object]
    ],
    probability_column: str,
) -> dict[str, float]:
    targets = [
        int(
            row[
                "target"
            ]
        )
        for row in rows
    ]

    probabilities = [
        float(
            row[
                probability_column
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
        for value in values
    ]

    gamma0 = (
        sum(
            value * value
            for value in centered
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
                for t in range(
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
    rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    logloss_difference = []

    brier_difference = []

    for row in rows:
        target = int(
            row[
                "target"
            ]
        )

        price = float(
            row[
                "mprice_warm_probability"
            ]
        )

        combined = float(
            row[
                "mcombined_warm_probability"
            ]
        )

        logloss_difference.append(
            observation_log_loss(
                target,
                combined,
            )
            - observation_log_loss(
                target,
                price,
            )
        )

        brier_difference.append(
            observation_brier(
                target,
                combined,
            )
            - observation_brier(
                target,
                price,
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

        results = {}

        for lag in HAC_LAGS:
            se = hac_standard_error(
                values,
                lag,
            )

            statistic = (
                difference_mean
                / se
                if se > 0.0
                else 0.0
            )

            results[
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
                results,
        }

    return output


def main() -> None:
    examples, feature_names = (
        load_examples()
    )

    calibration_stream = (
        build_calibration_stream(
            examples
        )
    )

    outer = load_outer_predictions()

    grouped = defaultdict(
        list
    )

    for row in outer:
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
            f"Expected 20 outer folds, "
            f"got {len(folds)}."
        )

    output_rows = []

    alpha_rows = []

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

        history = [
            record
            for record
            in calibration_stream
            if (
                record[
                    "label_available_time"
                ]
                <= test_start
            )
        ]

        if not history:
            raise RuntimeError(
                f"No legal calibration history "
                f"for outer fold {fold}."
            )

        price_alpha, price_score = (
            choose_alpha(
                history,
                "mprice_probability",
            )
        )

        (
            combined_alpha,
            combined_score,
        ) = choose_alpha(
            history,
            "mcombined_probability",
        )

        alpha_rows.append(
            {
                "fold":
                    fold,
                "test_start":
                    test_start.isoformat(),
                "calibration_observations":
                    len(
                        history
                    ),
                "price_alpha":
                    price_alpha,
                "price_selection_log_loss":
                    price_score,
                "combined_alpha":
                    combined_alpha,
                "combined_selection_log_loss":
                    combined_score,
            }
        )

        for row in fold_rows:
            price_warm = (
                shrink_probability(
                    float(
                        row[
                            "m0_probability"
                        ]
                    ),
                    float(
                        row[
                            "mprice_probability"
                        ]
                    ),
                    price_alpha,
                )
            )

            combined_warm = (
                shrink_probability(
                    float(
                        row[
                            "m0_probability"
                        ]
                    ),
                    float(
                        row[
                            "mcombined_probability"
                        ]
                    ),
                    combined_alpha,
                )
            )

            output_rows.append(
                {
                    **row,
                    "decision_time":
                        row[
                            "decision_time"
                        ].isoformat(),
                    "mprice_warm_alpha":
                        price_alpha,
                    "mprice_warm_probability":
                        price_warm,
                    "mcombined_warm_alpha":
                        combined_alpha,
                    "mcombined_warm_probability":
                        combined_warm,
                }
            )

    result_metrics = {
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
        "M_PRICE_WARM":
            metrics(
                output_rows,
                "mprice_warm_probability",
            ),
        "M_COMBINED_RAW":
            metrics(
                output_rows,
                "mcombined_probability",
            ),
        "M_COMBINED_WARM":
            metrics(
                output_rows,
                "mcombined_warm_probability",
            ),
    }

    fold_wins = {
        "log_loss":
            0,
        "brier":
            0,
        "accuracy":
            0,
    }

    fold_deltas = []

    for fold in folds:
        rows = [
            row
            for row
            in output_rows
            if int(
                row[
                    "fold"
                ]
            ) == fold
        ]

        price = metrics(
            rows,
            "mprice_warm_probability",
        )

        combined = metrics(
            rows,
            "mcombined_warm_probability",
        )

        ll_delta = (
            combined[
                "log_loss"
            ]
            - price[
                "log_loss"
            ]
        )

        brier_delta = (
            combined[
                "brier_score"
            ]
            - price[
                "brier_score"
            ]
        )

        accuracy_delta = (
            combined[
                "accuracy"
            ]
            - price[
                "accuracy"
            ]
        )

        fold_wins[
            "log_loss"
        ] += int(
            ll_delta < 0.0
        )

        fold_wins[
            "brier"
        ] += int(
            brier_delta < 0.0
        )

        fold_wins[
            "accuracy"
        ] += int(
            accuracy_delta > 0.0
        )

        fold_deltas.append(
            {
                "fold":
                    fold,
                "log_loss_delta_combined_minus_price":
                    ll_delta,
                "brier_delta_combined_minus_price":
                    brier_delta,
                "accuracy_delta_combined_minus_price":
                    accuracy_delta,
            }
        )

    significance = hac_comparison(
        output_rows
    )

    results = {
        "experiment":
            "warm-start-probability-shrinkage-4h",
        "calibration_protocol": {
            "type":
                "strict_prequential_historical_walk_forward",
            "minimum_training_observations":
                CAL_MIN_TRAIN,
            "test_observations":
                CAL_TEST_OBSERVATIONS,
            "outer_information_rule":
                (
                    "Only calibration forecasts whose "
                    "realized labels were available by "
                    "the outer test-start time are used "
                    "to select alpha."
                ),
            "alpha_grid_min":
                0.0,
            "alpha_grid_max":
                1.0,
            "alpha_grid_step":
                0.01,
            "selection_metric":
                "log_loss",
        },
        "combined_feature_count":
            len(
                feature_names
            ),
        "price_feature_count":
            len(
                PRICE_FEATURES
            ),
        "calibration_stream_forecasts":
            len(
                calibration_stream
            ),
        "outer_oos_forecasts":
            len(
                output_rows
            ),
        "metrics":
            result_metrics,
        "warm_combined_minus_warm_price": {
            "log_loss_delta":
                (
                    result_metrics[
                        "M_COMBINED_WARM"
                    ][
                        "log_loss"
                    ]
                    - result_metrics[
                        "M_PRICE_WARM"
                    ][
                        "log_loss"
                    ]
                ),
            "brier_delta":
                (
                    result_metrics[
                        "M_COMBINED_WARM"
                    ][
                        "brier_score"
                    ]
                    - result_metrics[
                        "M_PRICE_WARM"
                    ][
                        "brier_score"
                    ]
                ),
            "accuracy_delta":
                (
                    result_metrics[
                        "M_COMBINED_WARM"
                    ][
                        "accuracy"
                    ]
                    - result_metrics[
                        "M_PRICE_WARM"
                    ][
                        "accuracy"
                    ]
                ),
        },
        "combined_fold_wins":
            fold_wins,
        "hac_significance":
            significance,
        "alpha_schedule":
            alpha_rows,
        "fold_deltas":
            fold_deltas,
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

    with CALIBRATION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "calibration_fold",
            "decision_time",
            "label_available_time",
            "target",
            "m0_probability",
            "mprice_probability",
            "mcombined_probability",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in calibration_stream:
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

    with ALPHA_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                alpha_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            alpha_rows
        )

    print(
        "WARM-START PROBABILITY SHRINKAGE"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Calibration forecasts: "
        f"{len(calibration_stream):,}"
    )

    print(
        f"Outer OOS forecasts: "
        f"{len(output_rows):,}"
    )

    print()

    for name in (
        "M0",
        "M_PRICE_RAW",
        "M_PRICE_WARM",
        "M_COMBINED_RAW",
        "M_COMBINED_WARM",
    ):
        item = result_metrics[
            name
        ]

        print(
            f"{name:17s} "
            f"LL={item['log_loss']:.8f}  "
            f"Brier="
            f"{item['brier_score']:.8f}  "
            f"Acc="
            f"{item['accuracy']:.4%}"
        )

    comparison = results[
        "warm_combined_minus_warm_price"
    ]

    print()
    print(
        "WARM COMBINED - WARM PRICE"
    )
    print(
        "--------------------------"
    )

    print(
        "Log loss: "
        f"{comparison['log_loss_delta']:+.8f}"
    )

    print(
        "Brier:    "
        f"{comparison['brier_delta']:+.8f}"
    )

    print(
        "Accuracy: "
        f"{comparison['accuracy_delta']:+.4%}"
    )

    print(
        "Fold wins combined: "
        f"LL {fold_wins['log_loss']}/20, "
        f"Brier {fold_wins['brier']}/20, "
        f"Acc {fold_wins['accuracy']}/20"
    )

    print()

    for metric in (
        "log_loss",
        "brier",
    ):
        item = significance[
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
        "ALPHA SCHEDULE"
    )
    print(
        "--------------"
    )

    for row in alpha_rows:
        print(
            f"Fold {row['fold']:2d}: "
            f"n_cal="
            f"{row['calibration_observations']:5d}, "
            f"price="
            f"{row['price_alpha']:.2f}, "
            f"combined="
            f"{row['combined_alpha']:.2f}"
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
        "Calibration ->",
        CALIBRATION_FILE,
    )

    print(
        "Alphas      ->",
        ALPHA_FILE,
    )


if __name__ == "__main__":
    main()
