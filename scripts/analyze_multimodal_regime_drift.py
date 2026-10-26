from __future__ import annotations

import csv
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_nested_latent_neural_walk_forward as base

from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
)
from latent_information_dynamics.models.latent.regime_drift import (
    covariance_shift,
    log_scale_shift,
    standardized_mean_shift,
)


NESTED_RESULTS_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward/"
    "nested_latent_neural_walk_forward_4h_2020_2025.json"
)

HORIZON_FOLD_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/training_horizon/"
    "multimodal_training_horizon_folds.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/regime_drift"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "multimodal_regime_drift.json"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "multimodal_regime_drift_folds.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "multimodal_regime_drift_correlations.csv"
)


def modality_matrix(
    examples: list[
        base.Example
    ],
    *,
    modality: str,
) -> np.ndarray:
    if modality == "price":
        values = [
            example.price_x
            for example
            in examples
        ]

    elif modality == "flow":
        values = [
            example.flow_x
            for example
            in examples
        ]

    else:
        raise ValueError(
            "Unknown modality."
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def modality_diagnostics(
    validation: list[
        base.Example
    ],
    test: list[
        base.Example
    ],
    *,
    modality: str,
) -> dict[str, float]:
    validation_matrix = (
        modality_matrix(
            validation,
            modality=modality,
        )
    )

    test_matrix = (
        modality_matrix(
            test,
            modality=modality,
        )
    )

    mean_shift = (
        standardized_mean_shift(
            validation_matrix,
            test_matrix,
        )
    )

    scale_shift = (
        log_scale_shift(
            validation_matrix,
            test_matrix,
        )
    )

    covariance = (
        covariance_shift(
            validation_matrix,
            test_matrix,
        )
    )

    return {
        "mean_shift_mean_abs":
            mean_shift[
                "mean_absolute"
            ],
        "mean_shift_rms":
            mean_shift[
                "rms"
            ],
        "mean_shift_max":
            mean_shift[
                "maximum_absolute"
            ],
        "scale_shift_mean_abs":
            scale_shift[
                "mean_absolute"
            ],
        "scale_shift_rms":
            scale_shift[
                "rms"
            ],
        "scale_shift_max":
            scale_shift[
                "maximum_absolute"
            ],
        "covariance_shift":
            covariance,
    }


def write_csv(
    path: Path,
    rows: list[
        dict[str, object]
    ],
) -> None:
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

        writer.writerows(
            rows
        )


def main() -> None:
    nested_results = json.loads(
        NESTED_RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    (
        examples,
        _,
    ) = base.load_examples()

    baseline_rows = (
        base.load_baseline_rows()
    )

    example_by_time = {
        example.decision_time:
            example
        for example
        in examples
    }

    grouped = defaultdict(
        list
    )

    for row in baseline_rows:
        grouped[
            int(
                row["fold"]
            )
        ].append(
            row
        )

    nested_fold_by_id = {
        int(
            row["fold"]
        ):
            row
        for row
        in nested_results[
            "folds"
        ]
    }

    records = []

    for fold in range(
        20
    ):
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

        validation = (
            outer_train[
                -base.INNER_VALIDATION_OBSERVATIONS:
            ]
        )

        outer_test = [
            example_by_time[
                row[
                    "decision_time"
                ]
            ]
            for row
            in frozen
        ]

        price = (
            modality_diagnostics(
                validation,
                outer_test,
                modality="price",
            )
        )

        flow = (
            modality_diagnostics(
                validation,
                outer_test,
                modality="flow",
            )
        )

        nested = (
            nested_fold_by_id[
                fold
            ]
        )

        validation_up_share = float(
            np.mean(
                [
                    example.target
                    for example
                    in validation
                ]
            )
        )

        test_up_share = float(
            np.mean(
                [
                    example.target
                    for example
                    in outer_test
                ]
            )
        )

        record = {
            "fold":
                fold,
            "selected_epoch":
                int(
                    nested[
                        "multimodal_selected_epochs"
                    ]
                ),
            "multimodal_minus_price_nn":
                float(
                    nested[
                        "multimodal_minus_price_nn"
                    ]
                ),
            "validation_up_share":
                validation_up_share,
            "test_up_share":
                test_up_share,
            "absolute_target_share_shift":
                abs(
                    test_up_share
                    - validation_up_share
                ),
        }

        for key, value in (
            price.items()
        ):
            record[
                f"price_{key}"
            ] = value

        for key, value in (
            flow.items()
        ):
            record[
                f"flow_{key}"
            ] = value

        records.append(
            record
        )

    frame = pd.DataFrame(
        records
    )

    horizon_available = (
        HORIZON_FOLD_FILE.exists()
    )

    if horizon_available:
        horizon = pd.read_csv(
            HORIZON_FOLD_FILE
        )

        horizon[
            "fold"
        ] = (
            horizon[
                "fold"
            ]
            .astype(
                int
            )
        )

        keep = [
            "fold",
        ]

        for epoch in (
            1,
            2,
            5,
            10,
            20,
            50,
        ):
            column = (
                f"epoch_{epoch:02d}_minus_selected"
            )

            if column in horizon.columns:
                keep.append(
                    column
                )

        frame = frame.merge(
            horizon[
                keep
            ],
            on="fold",
            how="left",
            validate="one_to_one",
        )

    drift_columns = [
        column
        for column
        in frame.columns
        if (
            column.startswith(
                "price_"
            )
            or column.startswith(
                "flow_"
            )
            or column
            == "absolute_target_share_shift"
        )
    ]

    target = frame[
        "multimodal_minus_price_nn"
    ].to_numpy(
        dtype=np.float64
    )

    correlation_rows = []

    for column in drift_columns:
        values = frame[
            column
        ].to_numpy(
            dtype=np.float64
        )

        correlation_rows.append(
            {
                "diagnostic":
                    column,
                "target":
                    "multimodal_minus_price_nn",
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

    if horizon_available:
        rescue_columns = [
            column
            for column
            in frame.columns
            if column.endswith(
                "_minus_selected"
            )
        ]

        for rescue_column in (
            rescue_columns
        ):
            # Negative fixed-minus-selected means
            # the fixed shorter/longer horizon improved
            # on the selected model.
            repair = (
                -frame[
                    rescue_column
                ].to_numpy(
                    dtype=np.float64
                )
            )

            for drift_column in (
                drift_columns
            ):
                values = frame[
                    drift_column
                ].to_numpy(
                    dtype=np.float64
                )

                correlation_rows.append(
                    {
                        "diagnostic":
                            drift_column,
                        "target":
                            (
                                rescue_column
                                + "_repair"
                            ),
                        "pearson":
                            correlation(
                                values,
                                repair,
                                method="pearson",
                            ),
                        "spearman":
                            correlation(
                                values,
                                repair,
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

    write_csv(
        FOLD_FILE,
        frame.to_dict(
            orient="records"
        ),
    )

    correlations.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    results = {
        "experiment":
            "multimodal-validation-to-oos-regime-drift",
        "interpretation":
            (
                "Post-hoc developmental diagnostic. "
                "Drift measures compare the exact inner "
                "validation block used for epoch selection "
                "with the subsequent untouched outer-test "
                "block."
            ),
        "folds":
            20,
        "horizon_ablation_available":
            horizon_available,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "MULTIMODAL REGIME-DRIFT DIAGNOSTICS"
    )

    print(
        "=================================="
    )

    print()
    print(
        "DRIFT VS MULTIMODAL OOS FAILURE"
    )

    print(
        "-------------------------------"
    )

    failure_correlations = (
        correlations[
            correlations[
                "target"
            ]
            == "multimodal_minus_price_nn"
        ]
        .copy()
    )

    failure_correlations[
        "absolute_spearman"
    ] = (
        failure_correlations[
            "spearman"
        ].abs()
    )

    failure_correlations = (
        failure_correlations
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
    )

    for _, row in (
        failure_correlations
        .head(
            12
        )
        .iterrows()
    ):
        print(
            f"{row['diagnostic']:34s} "
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

    worst = (
        frame
        .sort_values(
            "multimodal_minus_price_nn",
            ascending=False,
        )
        .head(
            7
        )
    )

    for _, row in (
        worst.iterrows()
    ):
        print(
            f"Fold {int(row['fold']):02d} "
            f"gap="
            f"{row['multimodal_minus_price_nn']:+.6f} "
            f"epoch="
            f"{int(row['selected_epoch']):2d} "
            f"PriceMeanDrift="
            f"{row['price_mean_shift_rms']:.3f} "
            f"FlowMeanDrift="
            f"{row['flow_mean_shift_rms']:.3f} "
            f"PriceCovDrift="
            f"{row['price_covariance_shift']:.3f} "
            f"FlowCovDrift="
            f"{row['flow_covariance_shift']:.3f}"
        )

    print()
    print(
        "HIGHEST FLOW-DRIFT FOLDS"
    )

    print(
        "------------------------"
    )

    highest_flow = (
        frame
        .sort_values(
            "flow_mean_shift_rms",
            ascending=False,
        )
        .head(
            7
        )
    )

    for _, row in (
        highest_flow.iterrows()
    ):
        print(
            f"Fold {int(row['fold']):02d} "
            f"FlowMeanDrift="
            f"{row['flow_mean_shift_rms']:.3f} "
            f"FlowScaleDrift="
            f"{row['flow_scale_shift_rms']:.3f} "
            f"FlowCovDrift="
            f"{row['flow_covariance_shift']:.3f} "
            f"OOSgap="
            f"{row['multimodal_minus_price_nn']:+.6f}"
        )

    if horizon_available:
        print()
        print(
            "DRIFT VS SHORT-HORIZON REPAIR"
        )

        print(
            "-----------------------------"
        )

        repair_correlations = (
            correlations[
                correlations[
                    "target"
                ].str.endswith(
                    "_repair"
                )
            ]
            .copy()
        )

        repair_correlations[
            "absolute_spearman"
        ] = (
            repair_correlations[
                "spearman"
            ].abs()
        )

        repair_correlations = (
            repair_correlations
            .sort_values(
                "absolute_spearman",
                ascending=False,
            )
        )

        for _, row in (
            repair_correlations
            .head(
                15
            )
            .iterrows()
        ):
            print(
                f"{row['target']:31s} "
                f"{row['diagnostic']:32s} "
                f"P="
                f"{row['pearson']:+.3f} "
                f"S="
                f"{row['spearman']:+.3f}"
            )

    print()
    print(
        "Results      ->",
        RESULTS_FILE,
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
