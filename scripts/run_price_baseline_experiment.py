from __future__ import annotations

import csv
import hashlib
import json

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from latent_information_dynamics.datasets.supervised import (
    SupervisedDataset,
    SupervisedExample,
)
from latent_information_dynamics.evaluation.probability import (
    brier_score,
    calibration_bins,
    classification_accuracy,
    log_loss,
)
from latent_information_dynamics.evaluation.walk_forward import (
    expanding_walk_forward_folds,
    evaluate_walk_forward,
)
from latent_information_dynamics.models.binary import (
    ConstantProbabilityModel,
    LogisticRegressionBinary,
)


DATASET_FILE = Path(
    "data/derived/supervised/"
    "btc_price_supervised_4h_2020_2025.csv"
)

DATASET_METADATA_FILE = Path(
    "data/derived/supervised/"
    "btc_price_supervised_4h_2020_2025.json"
)

EXPECTED_DATASET_SHA256 = (
    "7b6fd7f35cab32adb0e9eeab5be27a05"
    "db18c6217990239d10c32a4fd49997cc"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "price_baseline"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "m0_vs_mprice_4h_2020_2025.json"
)

PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "m0_vs_mprice_predictions.csv"
)

FOLDS_FILE = (
    OUTPUT_ROOT
    / "m0_vs_mprice_folds.csv"
)

MIN_TRAIN_OBSERVATIONS = 2190
TEST_OBSERVATIONS = 546
LOGISTIC_L2 = 1.0
N_CALIBRATION_BINS = 10


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_dataset() -> SupervisedDataset:
    observed_hash = sha256_file(
        DATASET_FILE
    )

    if (
        observed_hash
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Frozen supervised dataset "
            "hash does not match the "
            "Day 35 dataset."
        )

    metadata = json.loads(
        DATASET_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    feature_names = tuple(
        metadata["features"]
    )

    if len(feature_names) != 11:
        raise RuntimeError(
            "Expected 11 price features."
        )

    examples = []

    with DATASET_FILE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            examples.append(
                SupervisedExample(
                    decision_time=
                        datetime.fromisoformat(
                            row[
                                "decision_time"
                            ]
                        ),
                    resolution_time=
                        datetime.fromisoformat(
                            row[
                                "resolution_time"
                            ]
                        ),
                    label_available_time=
                        datetime.fromisoformat(
                            row[
                                "label_available_time"
                            ]
                        ),
                    feature_names=
                        feature_names,
                    feature_values=
                        tuple(
                            float(
                                row[name]
                            )
                            for name
                            in feature_names
                        ),
                    target=int(
                        row["target"]
                    ),
                    realized_return=float(
                        row[
                            "realized_return"
                        ]
                    ),
                    realized_log_return=float(
                        row[
                            "realized_log_return"
                        ]
                    ),
                )
            )

    dataset = SupervisedDataset(
        examples=tuple(examples)
    )

    if len(dataset.examples) != 13030:
        raise RuntimeError(
            "Expected exactly 13,030 "
            "frozen examples."
        )

    return dataset


def grouped_forecasts(
    result,
):
    grouped = defaultdict(
        list
    )

    for forecast in result.forecasts:
        grouped[
            forecast.fold
        ].append(
            forecast
        )

    return grouped


def calibration_records(
    result,
):
    return [
        {
            "lower":
                item.lower,
            "upper":
                item.upper,
            "count":
                item.count,
            "mean_probability":
                item.mean_probability,
            "event_rate":
                item.event_rate,
        }
        for item in calibration_bins(
            result.targets,
            result.probabilities,
            n_bins=N_CALIBRATION_BINS,
        )
    ]


def main() -> None:
    print(
        "Loading frozen Day 35 "
        "supervised dataset..."
    )

    dataset = load_dataset()

    print(
        f"Rows: {len(dataset.examples):,}"
    )

    print(
        f"Features: "
        f"{len(dataset.feature_names)}"
    )

    folds = (
        expanding_walk_forward_folds(
            dataset,
            min_train_observations=
                MIN_TRAIN_OBSERVATIONS,
            test_observations=
                TEST_OBSERVATIONS,
        )
    )

    print(
        f"Walk-forward folds: "
        f"{len(folds)}"
    )

    print()
    print(
        "Evaluating M0 "
        "(historical unconditional probability)..."
    )

    m0 = evaluate_walk_forward(
        dataset,
        feature_names=
            dataset.feature_names,
        model_factory=
            ConstantProbabilityModel,
        min_train_observations=
            MIN_TRAIN_OBSERVATIONS,
        test_observations=
            TEST_OBSERVATIONS,
    )

    print(
        "Evaluating M_PRICE "
        "(L2 logistic price baseline)..."
    )

    mprice = evaluate_walk_forward(
        dataset,
        feature_names=
            dataset.feature_names,
        model_factory=lambda:
            LogisticRegressionBinary(
                l2=LOGISTIC_L2,
            ),
        min_train_observations=
            MIN_TRAIN_OBSERVATIONS,
        test_observations=
            TEST_OBSERVATIONS,
    )

    if (
        m0.decision_times
        != mprice.decision_times
    ):
        raise RuntimeError(
            "Models were not evaluated "
            "on identical timestamps."
        )

    if (
        m0.targets
        != mprice.targets
    ):
        raise RuntimeError(
            "Models were not evaluated "
            "against identical targets."
        )

    if not m0.forecasts:
        raise RuntimeError(
            "No OOS forecasts produced."
        )

    m0_by_fold = grouped_forecasts(
        m0
    )

    mprice_by_fold = (
        grouped_forecasts(
            mprice
        )
    )

    fold_rows = []

    price_logloss_wins = 0
    price_brier_wins = 0
    price_accuracy_wins = 0

    fold_logloss_deltas = []
    fold_brier_deltas = []
    fold_accuracy_deltas = []

    for fold in folds:
        baseline_forecasts = (
            m0_by_fold[
                fold.fold
            ]
        )

        price_forecasts = (
            mprice_by_fold[
                fold.fold
            ]
        )

        baseline_y = tuple(
            item.target
            for item
            in baseline_forecasts
        )

        baseline_p = tuple(
            item.probability
            for item
            in baseline_forecasts
        )

        price_y = tuple(
            item.target
            for item
            in price_forecasts
        )

        price_p = tuple(
            item.probability
            for item
            in price_forecasts
        )

        if baseline_y != price_y:
            raise RuntimeError(
                "Fold target mismatch."
            )

        m0_ll = log_loss(
            baseline_y,
            baseline_p,
        )

        mp_ll = log_loss(
            price_y,
            price_p,
        )

        m0_bs = brier_score(
            baseline_y,
            baseline_p,
        )

        mp_bs = brier_score(
            price_y,
            price_p,
        )

        m0_acc = (
            classification_accuracy(
                baseline_y,
                baseline_p,
            )
        )

        mp_acc = (
            classification_accuracy(
                price_y,
                price_p,
            )
        )

        ll_delta = (
            mp_ll - m0_ll
        )

        bs_delta = (
            mp_bs - m0_bs
        )

        acc_delta = (
            mp_acc - m0_acc
        )

        fold_logloss_deltas.append(
            ll_delta
        )

        fold_brier_deltas.append(
            bs_delta
        )

        fold_accuracy_deltas.append(
            acc_delta
        )

        if ll_delta < 0:
            price_logloss_wins += 1

        if bs_delta < 0:
            price_brier_wins += 1

        if acc_delta > 0:
            price_accuracy_wins += 1

        fold_rows.append(
            {
                "fold":
                    fold.fold,
                "training_cutoff":
                    fold.training_cutoff.isoformat(),
                "train_observations":
                    len(
                        fold.train_indices
                    ),
                "test_observations":
                    len(
                        fold.test_indices
                    ),
                "test_start":
                    dataset.examples[
                        fold.test_indices[
                            0
                        ]
                    ].decision_time.isoformat(),
                "test_end":
                    dataset.examples[
                        fold.test_indices[
                            -1
                        ]
                    ].decision_time.isoformat(),
                "m0_log_loss":
                    m0_ll,
                "mprice_log_loss":
                    mp_ll,
                "log_loss_delta":
                    ll_delta,
                "m0_brier":
                    m0_bs,
                "mprice_brier":
                    mp_bs,
                "brier_delta":
                    bs_delta,
                "m0_accuracy":
                    m0_acc,
                "mprice_accuracy":
                    mp_acc,
                "accuracy_delta":
                    acc_delta,
            }
        )

    log_loss_delta = (
        mprice.log_loss
        - m0.log_loss
    )

    brier_delta = (
        mprice.brier_score
        - m0.brier_score
    )

    accuracy_delta = (
        mprice.accuracy
        - m0.accuracy
    )

    relative_log_loss_improvement = (
        (
            m0.log_loss
            - mprice.log_loss
        )
        / m0.log_loss
    )

    relative_brier_improvement = (
        (
            m0.brier_score
            - mprice.brier_score
        )
        / m0.brier_score
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "decision_time",
                "target",
                "m0_probability",
                "mprice_probability",
                "probability_difference",
            ],
        )

        writer.writeheader()

        for baseline, price in zip(
            m0.forecasts,
            mprice.forecasts,
            strict=True,
        ):
            writer.writerow(
                {
                    "fold":
                        baseline.fold,
                    "decision_time":
                        baseline.decision_time.isoformat(),
                    "target":
                        baseline.target,
                    "m0_probability":
                        baseline.probability,
                    "mprice_probability":
                        price.probability,
                    "probability_difference":
                        (
                            price.probability
                            - baseline.probability
                        ),
                }
            )

    with FOLDS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fold_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            fold_rows
        )

    results = {
        "experiment":
            "m0-vs-mprice-4h",
        "dataset":
            "btc-price-supervised-4h-2020-2025",
        "dataset_sha256":
            EXPECTED_DATASET_SHA256,
        "dataset_rows":
            len(dataset.examples),
        "features":
            list(
                dataset.feature_names
            ),
        "protocol": {
            "type":
                "strict_expanding_walk_forward",
            "min_train_observations":
                MIN_TRAIN_OBSERVATIONS,
            "test_observations":
                TEST_OBSERVATIONS,
            "folds":
                len(folds),
            "oos_forecasts":
                len(m0.forecasts),
            "first_oos_decision":
                m0.decision_times[
                    0
                ].isoformat(),
            "last_oos_decision":
                m0.decision_times[
                    -1
                ].isoformat(),
            "logistic_l2":
                LOGISTIC_L2,
            "hyperparameter_search":
                False,
        },
        "m0": {
            "description":
                "Historical unconditional "
                "Bernoulli probability "
                "estimated from legally "
                "available training labels.",
            "log_loss":
                m0.log_loss,
            "brier_score":
                m0.brier_score,
            "accuracy":
                m0.accuracy,
            "mean_probability":
                mean(
                    m0.probabilities
                ),
            "calibration":
                calibration_records(
                    m0
                ),
        },
        "mprice": {
            "description":
                "L2-regularized logistic "
                "regression on 11 historical "
                "BTC price features.",
            "log_loss":
                mprice.log_loss,
            "brier_score":
                mprice.brier_score,
            "accuracy":
                mprice.accuracy,
            "mean_probability":
                mean(
                    mprice.probabilities
                ),
            "calibration":
                calibration_records(
                    mprice
                ),
        },
        "comparison": {
            "log_loss_delta_mprice_minus_m0":
                log_loss_delta,
            "relative_log_loss_improvement":
                relative_log_loss_improvement,
            "brier_delta_mprice_minus_m0":
                brier_delta,
            "relative_brier_improvement":
                relative_brier_improvement,
            "accuracy_delta_mprice_minus_m0":
                accuracy_delta,
            "mprice_logloss_fold_wins":
                price_logloss_wins,
            "mprice_brier_fold_wins":
                price_brier_wins,
            "mprice_accuracy_fold_wins":
                price_accuracy_wins,
            "mean_fold_logloss_delta":
                mean(
                    fold_logloss_deltas
                ),
            "mean_fold_brier_delta":
                mean(
                    fold_brier_deltas
                ),
            "mean_fold_accuracy_delta":
                mean(
                    fold_accuracy_deltas
                ),
        },
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
        "M0 VS M_PRICE — OOS RESULTS"
    )
    print(
        "---------------------------"
    )
    print(
        f"OOS forecasts:       "
        f"{len(m0.forecasts):,}"
    )
    print(
        f"Folds:               "
        f"{len(folds)}"
    )
    print()
    print(
        f"M0 log loss:         "
        f"{m0.log_loss:.8f}"
    )
    print(
        f"M_PRICE log loss:    "
        f"{mprice.log_loss:.8f}"
    )
    print(
        f"Delta:               "
        f"{log_loss_delta:+.8f}"
    )
    print(
        f"Relative improvement:"
        f" "
        f"{relative_log_loss_improvement:.4%}"
    )
    print()
    print(
        f"M0 Brier:            "
        f"{m0.brier_score:.8f}"
    )
    print(
        f"M_PRICE Brier:       "
        f"{mprice.brier_score:.8f}"
    )
    print(
        f"Delta:               "
        f"{brier_delta:+.8f}"
    )
    print(
        f"Relative improvement:"
        f" "
        f"{relative_brier_improvement:.4%}"
    )
    print()
    print(
        f"M0 accuracy:         "
        f"{m0.accuracy:.4%}"
    )
    print(
        f"M_PRICE accuracy:    "
        f"{mprice.accuracy:.4%}"
    )
    print(
        f"Accuracy delta:      "
        f"{accuracy_delta:+.4%}"
    )
    print()
    print(
        f"M_PRICE fold wins "
        f"(log loss): "
        f"{price_logloss_wins}/"
        f"{len(folds)}"
    )
    print(
        f"M_PRICE fold wins "
        f"(Brier):    "
        f"{price_brier_wins}/"
        f"{len(folds)}"
    )
    print(
        f"M_PRICE fold wins "
        f"(accuracy): "
        f"{price_accuracy_wins}/"
        f"{len(folds)}"
    )
    print()
    print(
        f"Results     -> "
        f"{RESULTS_FILE}"
    )
    print(
        f"Predictions -> "
        f"{PREDICTIONS_FILE}"
    )
    print(
        f"Folds       -> "
        f"{FOLDS_FILE}"
    )


if __name__ == "__main__":
    main()
