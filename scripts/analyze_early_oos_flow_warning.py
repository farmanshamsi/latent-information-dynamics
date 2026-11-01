from __future__ import annotations

import csv
import json

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


LEAD_WINDOWS = (
    42,   # 7 days at 4-hour frequency
    84,   # 14 days
)

INTERVENTION_PREDICTIONS_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/modality_intervention/"
    "multimodal_modality_intervention_predictions.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/early_oos_warning"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "early_oos_flow_warning.json"
)

FOLD_FILE = (
    OUTPUT_ROOT
    / "early_oos_flow_warning_folds.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "early_oos_flow_warning_correlations.csv"
)


def feature_matrix(
    examples: list[base.Example],
    *,
    modality: str,
) -> np.ndarray:
    if modality == "flow":
        values = [
            example.flow_x
            for example in examples
        ]

    elif modality == "price":
        values = [
            example.price_x
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
    reference_matrix = feature_matrix(
        reference,
        modality=modality,
    )

    current_matrix = feature_matrix(
        current,
        modality=modality,
    )

    mean_shift = standardized_mean_shift(
        reference_matrix,
        current_matrix,
    )

    scale_shift = log_scale_shift(
        reference_matrix,
        current_matrix,
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
            covariance_shift(
                reference_matrix,
                current_matrix,
            ),
    }


def probability_metrics(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    return base.metric_set(
        frame[
            "target"
        ]
        .astype(int)
        .tolist(),
        frame[
            column
        ]
        .astype(float)
        .tolist(),
    )


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
    if not INTERVENTION_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Run modality intervention first."
        )

    predictions = pd.read_csv(
        INTERVENTION_PREDICTIONS_FILE
    )

    (
        examples,
        _,
    ) = base.load_examples()

    example_by_iso = {
        example.decision_time.isoformat():
            example
        for example in examples
    }

    records = []

    print(
        "EARLY-OOS FLOW WARNING DIAGNOSTIC"
    )

    print(
        "================================="
    )

    print()
    print(
        "PIT DESIGN"
    )

    print(
        "----------"
    )

    print(
        "Detector uses OOS FEATURES only."
    )

    print(
        "Detector uses no OOS labels."
    )

    print(
        "Performance is evaluated only AFTER "
        "the detector window."
    )

    for lead in LEAD_WINDOWS:
        for fold in range(20):
            fold_predictions = (
                predictions[
                    predictions[
                        "fold"
                    ]
                    == fold
                ]
                .sort_values(
                    "decision_time"
                )
                .reset_index(
                    drop=True
                )
            )

            if (
                fold_predictions.shape[0]
                <= lead
            ):
                raise RuntimeError(
                    "Lead window leaves no "
                    f"evaluation suffix in fold {fold}."
                )

            outer_test_start = (
                base.parse_datetime(
                    str(
                        fold_predictions.iloc[
                            0
                        ][
                            "decision_time"
                        ]
                    )
                )
            )

            outer_train = [
                example
                for example in examples
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

            early_rows = (
                fold_predictions.iloc[
                    :lead
                ]
            )

            suffix = (
                fold_predictions.iloc[
                    lead:
                ]
            )

            early_examples = [
                example_by_iso[
                    str(
                        decision_time
                    )
                ]
                for decision_time
                in early_rows[
                    "decision_time"
                ]
            ]

            flow = drift_metrics(
                validation,
                early_examples,
                modality="flow",
            )

            price = drift_metrics(
                validation,
                early_examples,
                modality="price",
            )

            full = probability_metrics(
                suffix,
                "full_probability",
            )

            flow_neutral = (
                probability_metrics(
                    suffix,
                    "flow_neutral_probability",
                )
            )

            price_nn = probability_metrics(
                suffix,
                "price_nn_probability",
            )

            # Positive = removing Flow improves
            # subsequent OOS log loss.
            suffix_flow_harm = (
                full[
                    "log_loss"
                ]
                - flow_neutral[
                    "log_loss"
                ]
            )

            multimodal_gap = (
                full[
                    "log_loss"
                ]
                - price_nn[
                    "log_loss"
                ]
            )

            record = {
                "lead_observations":
                    lead,
                "lead_days":
                    lead / 6.0,
                "fold":
                    fold,
                "detector_start":
                    str(
                        early_rows.iloc[
                            0
                        ][
                            "decision_time"
                        ]
                    ),
                "evaluation_start":
                    str(
                        suffix.iloc[
                            0
                        ][
                            "decision_time"
                        ]
                    ),
                "evaluation_observations":
                    int(
                        suffix.shape[0]
                    ),
                "suffix_full_log_loss":
                    full[
                        "log_loss"
                    ],
                "suffix_flow_neutral_log_loss":
                    flow_neutral[
                        "log_loss"
                    ],
                "suffix_price_nn_log_loss":
                    price_nn[
                        "log_loss"
                    ],
                "suffix_flow_harm":
                    suffix_flow_harm,
                "suffix_multimodal_gap":
                    multimodal_gap,
            }

            for key, value in (
                flow.items()
            ):
                record[
                    f"flow_{key}"
                ] = value

            for key, value in (
                price.items()
            ):
                record[
                    f"price_{key}"
                ] = value

            records.append(
                record
            )

    frame = pd.DataFrame(
        records
    )

    diagnostics = [
        column
        for column in frame.columns
        if (
            column.startswith(
                "flow_"
            )
            or column.startswith(
                "price_"
            )
        )
    ]

    correlation_rows = []

    for lead in LEAD_WINDOWS:
        lead_frame = frame[
            frame[
                "lead_observations"
            ]
            == lead
        ]

        for target_name in (
            "suffix_flow_harm",
            "suffix_multimodal_gap",
        ):
            target = lead_frame[
                target_name
            ].to_numpy(
                dtype=np.float64
            )

            for diagnostic in diagnostics:
                values = lead_frame[
                    diagnostic
                ].to_numpy(
                    dtype=np.float64
                )

                correlation_rows.append(
                    {
                        "lead_observations":
                            lead,
                        "lead_days":
                            lead / 6.0,
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
            "early-oos-flow-warning",
        "status":
            (
                "Developmental diagnostic. "
                "Early OOS feature observations are "
                "used without labels to predict Flow "
                "harm over the remaining OOS block."
            ),
        "protocol": {
            "lead_windows":
                list(
                    LEAD_WINDOWS
                ),
            "lead_days":
                [
                    lead / 6.0
                    for lead
                    in LEAD_WINDOWS
                ],
            "reference":
                (
                    "Immediately preceding chronological "
                    "inner-validation block."
                ),
            "oos_labels_used_by_detector":
                False,
            "future_oos_features_used_by_detector":
                False,
            "evaluation_overlap_with_detector":
                False,
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

    for lead in LEAD_WINDOWS:
        print()
        print(
            f"{lead // 6}-DAY EARLY WARNING "
            "VS SUBSEQUENT FLOW HARM"
        )

        print(
            "-" * 49
        )

        subset = correlations[
            (
                correlations[
                    "lead_observations"
                ]
                == lead
            )
            & (
                correlations[
                    "target"
                ]
                == "suffix_flow_harm"
            )
        ].copy()

        subset[
            "absolute_spearman"
        ] = (
            subset[
                "spearman"
            ].abs()
        )

        subset = subset.sort_values(
            "absolute_spearman",
            ascending=False,
        )

        for _, row in (
            subset.head(
                12
            ).iterrows()
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
            f"{lead // 6}-DAY WARNING "
            "VS SUBSEQUENT MULTIMODAL FAILURE"
        )

        print(
            "-" * 52
        )

        subset = correlations[
            (
                correlations[
                    "lead_observations"
                ]
                == lead
            )
            & (
                correlations[
                    "target"
                ]
                == "suffix_multimodal_gap"
            )
        ].copy()

        subset[
            "absolute_spearman"
        ] = (
            subset[
                "spearman"
            ].abs()
        )

        subset = subset.sort_values(
            "absolute_spearman",
            ascending=False,
        )

        for _, row in (
            subset.head(
                10
            ).iterrows()
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
            f"MOST HARMFUL SUFFIXES AFTER "
            f"{lead // 6} DAYS"
        )

        print(
            "-" * 37
        )

        worst = (
            frame[
                frame[
                    "lead_observations"
                ]
                == lead
            ]
            .sort_values(
                "suffix_flow_harm",
                ascending=False,
            )
            .head(
                8
            )
        )

        for _, row in (
            worst.iterrows()
        ):
            print(
                f"Fold {int(row['fold']):02d} "
                f"FlowHarm="
                f"{row['suffix_flow_harm']:+.6f} "
                f"MultiGap="
                f"{row['suffix_multimodal_gap']:+.6f} "
                f"FlowMean="
                f"{row['flow_mean_shift_rms']:.3f} "
                f"FlowScale="
                f"{row['flow_scale_shift_rms']:.3f} "
                f"FlowCov="
                f"{row['flow_covariance_shift']:.3f}"
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
