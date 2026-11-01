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


INTERVENTION_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/modality_intervention/"
    "multimodal_modality_intervention_folds.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/pretest_regime_warning"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "pretest_regime_warning.json"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "pretest_regime_warning_folds.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "pretest_regime_warning_correlations.csv"
)


WINDOW = (
    base.INNER_VALIDATION_OBSERVATIONS
)


def feature_matrix(
    examples: list[base.Example],
    *,
    modality: str,
) -> np.ndarray:
    if modality == "price":
        values = [
            example.price_x
            for example in examples
        ]

    elif modality == "flow":
        values = [
            example.flow_x
            for example in examples
        ]

    else:
        raise ValueError(
            "Unknown modality."
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def drift_metrics(
    reference: list[base.Example],
    current: list[base.Example],
    *,
    modality: str,
) -> dict[str, float]:
    reference_matrix = (
        feature_matrix(
            reference,
            modality=modality,
        )
    )

    current_matrix = (
        feature_matrix(
            current,
            modality=modality,
        )
    )

    mean_shift = (
        standardized_mean_shift(
            reference_matrix,
            current_matrix,
        )
    )

    scale_shift = (
        log_scale_shift(
            reference_matrix,
            current_matrix,
        )
    )

    covariance = (
        covariance_shift(
            reference_matrix,
            current_matrix,
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
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            "Cannot write empty CSV."
        )

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
        writer.writerows(rows)


def main() -> None:
    if not INTERVENTION_FILE.exists():
        raise FileNotFoundError(
            "Run the modality intervention first."
        )

    intervention = pd.read_csv(
        INTERVENTION_FILE
    )

    if intervention.shape[0] != 20:
        raise RuntimeError(
            "Expected 20 intervention folds."
        )

    (
        examples,
        _,
    ) = base.load_examples()

    baseline_rows = (
        base.load_baseline_rows()
    )

    grouped = defaultdict(list)

    for row in baseline_rows:
        grouped[
            int(row["fold"])
        ].append(row)

    intervention_by_fold = {
        int(row["fold"]):
            row
        for _, row
        in intervention.iterrows()
    }

    records = []

    for fold in range(20):
        frozen = sorted(
            grouped[fold],
            key=lambda row:
                row["decision_time"],
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

        if len(outer_train) < (
            2 * WINDOW
        ):
            raise RuntimeError(
                "Insufficient history for matched "
                f"pre-test windows in fold {fold}."
            )

        validation = (
            outer_train[
                -WINDOW:
            ]
        )

        earlier = (
            outer_train[
                :-WINDOW
            ]
        )

        adjacent_reference = (
            earlier[
                -WINDOW:
            ]
        )

        expanding_reference = (
            earlier
        )

        row = intervention_by_fold[
            fold
        ]

        record = {
            "fold":
                fold,
            "outer_test_start":
                outer_test_start
                .isoformat(),
            "reference_observations":
                len(
                    adjacent_reference
                ),
            "validation_observations":
                len(
                    validation
                ),
            "expanding_reference_observations":
                len(
                    expanding_reference
                ),
            "full_minus_price_nn":
                float(
                    row[
                        "full_minus_price_nn"
                    ]
                ),
            "flow_neutral_minus_full":
                float(
                    row[
                        "flow_neutral_minus_full"
                    ]
                ),
            # Positive means removing Flow later
            # improved OOS performance, therefore
            # Flow was harmful in that OOS block.
            "flow_harm":
                -float(
                    row[
                        "flow_neutral_minus_full"
                    ]
                ),
        }

        for modality in (
            "price",
            "flow",
        ):
            adjacent = drift_metrics(
                adjacent_reference,
                validation,
                modality=modality,
            )

            expanding = drift_metrics(
                expanding_reference,
                validation,
                modality=modality,
            )

            for key, value in (
                adjacent.items()
            ):
                record[
                    f"{modality}_adjacent_{key}"
                ] = value

            for key, value in (
                expanding.items()
            ):
                record[
                    f"{modality}_expanding_{key}"
                ] = value

        records.append(record)

    frame = pd.DataFrame(
        records
    )

    diagnostic_columns = [
        column
        for column in frame.columns
        if (
            column.startswith(
                "price_adjacent_"
            )
            or column.startswith(
                "flow_adjacent_"
            )
            or column.startswith(
                "price_expanding_"
            )
            or column.startswith(
                "flow_expanding_"
            )
        )
    ]

    targets = (
        "flow_harm",
        "full_minus_price_nn",
    )

    correlation_rows = []

    for target_name in targets:
        target = frame[
            target_name
        ].to_numpy(
            dtype=np.float64
        )

        for diagnostic in (
            diagnostic_columns
        ):
            values = frame[
                diagnostic
            ].to_numpy(
                dtype=np.float64
            )

            correlation_rows.append(
                {
                    "target":
                        target_name,
                    "diagnostic":
                        diagnostic,
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
            "pretest-regime-warning",
        "status":
            (
                "Developmental point-in-time diagnostic. "
                "All drift inputs occur before the "
                "subsequent outer-test block."
            ),
        "protocol": {
            "adjacent_reference_observations":
                WINDOW,
            "validation_observations":
                WINDOW,
            "adjacent_comparison":
                (
                    "Immediately preceding matched "
                    "window versus validation window."
                ),
            "expanding_comparison":
                (
                    "All pre-validation legal history "
                    "versus validation window."
                ),
            "outer_test_features_used":
                False,
            "outer_test_labels_used_for_drift":
                False,
        },
        "folds":
            20,
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
        "PRE-TEST REGIME WARNING"
    )

    print(
        "======================="
    )

    print()
    print(
        "PIT AUDIT"
    )

    print(
        "---------"
    )

    print(
        "Outer-test features used in warning: NO"
    )

    print(
        "Outer-test labels used in warning:   NO"
    )

    print(
        "Reference window:",
        WINDOW,
    )

    print(
        "Validation window:",
        WINDOW,
    )

    print()
    print(
        "PRE-TEST DRIFT VS SUBSEQUENT FLOW HARM"
    )

    print(
        "--------------------------------------"
    )

    flow_harm = correlations[
        correlations[
            "target"
        ]
        == "flow_harm"
    ].copy()

    flow_harm[
        "absolute_spearman"
    ] = (
        flow_harm[
            "spearman"
        ].abs()
    )

    flow_harm = (
        flow_harm
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
    )

    for _, row in (
        flow_harm.head(
            16
        ).iterrows()
    ):
        print(
            f"{row['diagnostic']:43s} "
            f"Pearson="
            f"{row['pearson']:+.3f} "
            f"Spearman="
            f"{row['spearman']:+.3f}"
        )

    print()
    print(
        "PRE-TEST DRIFT VS MULTIMODAL FAILURE"
    )

    print(
        "------------------------------------"
    )

    failure = correlations[
        correlations[
            "target"
        ]
        == "full_minus_price_nn"
    ].copy()

    failure[
        "absolute_spearman"
    ] = (
        failure[
            "spearman"
        ].abs()
    )

    failure = failure.sort_values(
        "absolute_spearman",
        ascending=False,
    )

    for _, row in (
        failure.head(
            12
        ).iterrows()
    ):
        print(
            f"{row['diagnostic']:43s} "
            f"Pearson="
            f"{row['pearson']:+.3f} "
            f"Spearman="
            f"{row['spearman']:+.3f}"
        )

    print()
    print(
        "MOST HARMFUL FLOW REGIMES"
    )

    print(
        "-------------------------"
    )

    worst = frame.sort_values(
        "flow_harm",
        ascending=False,
    ).head(
        8
    )

    for _, row in (
        worst.iterrows()
    ):
        print(
            f"Fold {int(row['fold']):02d} "
            f"FlowHarm="
            f"{row['flow_harm']:+.6f} "
            f"MultiGap="
            f"{row['full_minus_price_nn']:+.6f} "
            f"AdjFlowMean="
            f"{row['flow_adjacent_mean_shift_rms']:.3f} "
            f"AdjFlowScale="
            f"{row['flow_adjacent_scale_shift_rms']:.3f} "
            f"AdjFlowCov="
            f"{row['flow_adjacent_covariance_shift']:.3f} "
            f"ExpFlowMean="
            f"{row['flow_expanding_mean_shift_rms']:.3f}"
        )

    print()
    print(
        "FLOW-HARM SIGN"
    )

    print(
        "--------------"
    )

    harmful = int(
        np.sum(
            frame[
                "flow_harm"
            ]
            > 0.0
        )
    )

    helpful = int(
        np.sum(
            frame[
                "flow_harm"
            ]
            < 0.0
        )
    )

    print(
        "Flow harmful folds:",
        harmful,
    )

    print(
        "Flow helpful folds:",
        helpful,
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
