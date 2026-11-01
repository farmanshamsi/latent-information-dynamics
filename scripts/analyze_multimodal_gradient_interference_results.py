from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
)


ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/gradient_interference"
)

RESULTS_FILE = (
    ROOT
    / "multimodal_gradient_interference.json"
)

STEP_FILE = (
    ROOT
    / "multimodal_gradient_interference_steps.csv"
)

FOLD_FILE = (
    ROOT
    / "multimodal_gradient_interference_folds.csv"
)

CORRELATION_FILE = (
    ROOT
    / "multimodal_gradient_interference_correlations.csv"
)

OUTPUT_FILE = (
    ROOT
    / "multimodal_gradient_interference_diagnostics.json"
)


def finite(
    values: np.ndarray,
) -> np.ndarray:
    return values[
        np.isfinite(
            values
        )
    ]


def descriptive(
    values: np.ndarray,
) -> dict[str, float]:
    values = finite(
        values
    )

    if values.size == 0:
        raise ValueError(
            "No finite observations."
        )

    return {
        "mean":
            float(
                np.mean(
                    values
                )
            ),
        "median":
            float(
                np.median(
                    values
                )
            ),
        "p10":
            float(
                np.quantile(
                    values,
                    0.10,
                )
            ),
        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),
    }


def group_summary(
    folds: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, object]:
    subset = folds.loc[
        mask
    ]

    if subset.empty:
        raise ValueError(
            "Empty fold group."
        )

    metrics = (
        "mean_price_flow_cosine",
        "median_price_flow_cosine",
        "negative_cosine_fraction",
        "strong_conflict_fraction",
        "mean_flow_price_norm_ratio",
        "early_mean_cosine",
        "late_mean_cosine",
        "late_minus_early_cosine",
    )

    result = {
        "folds":
            int(
                subset.shape[0]
            )
    }

    for metric in metrics:
        result[
            metric
        ] = float(
            subset[
                metric
            ].mean()
        )

    return result


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Run the gradient-interference "
            "experiment first."
        )

    if not STEP_FILE.exists():
        raise FileNotFoundError(
            f"Missing {STEP_FILE}"
        )

    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FOLD_FILE}"
        )

    results = json.loads(
        RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    steps = pd.read_csv(
        STEP_FILE
    )

    folds = pd.read_csv(
        FOLD_FILE
    )

    if folds.shape[0] != 20:
        raise RuntimeError(
            "Expected 20 folds."
        )

    reproduction_error = float(
        results[
            "protocol"
        ][
            "maximum_probability_reproduction_error"
        ]
    )

    cosine = finite(
        steps[
            "price_flow_cosine"
        ].to_numpy(
            dtype=np.float64
        )
    )

    progress_bins = (
        pd.cut(
            steps[
                "training_progress"
            ],
            bins=[
                0.0,
                0.25,
                0.50,
                0.75,
                1.0000001,
            ],
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ],
            include_lowest=True,
        )
    )

    steps = steps.copy()

    steps[
        "progress_quartile"
    ] = progress_bins

    quartile_rows = []

    for quartile in (
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ):
        subset = steps[
            steps[
                "progress_quartile"
            ]
            == quartile
        ]

        quartile_cosine = finite(
            subset[
                "price_flow_cosine"
            ].to_numpy(
                dtype=np.float64
            )
        )

        quartile_rows.append(
            {
                "quartile":
                    quartile,
                "steps":
                    int(
                        quartile_cosine.size
                    ),
                "mean_cosine":
                    float(
                        np.mean(
                            quartile_cosine
                        )
                    ),
                "median_cosine":
                    float(
                        np.median(
                            quartile_cosine
                        )
                    ),
                "negative_fraction":
                    float(
                        np.mean(
                            quartile_cosine
                            < 0.0
                        )
                    ),
                "strong_conflict_fraction":
                    float(
                        np.mean(
                            quartile_cosine
                            < -0.25
                        )
                    ),
            }
        )

    harmful_mask = (
        folds[
            "flow_harm"
        ].to_numpy(
            dtype=np.float64
        )
        > 0.0
    )

    helpful_mask = (
        folds[
            "flow_harm"
        ].to_numpy(
            dtype=np.float64
        )
        < 0.0
    )

    failure_mask = (
        folds[
            "multimodal_minus_price_nn"
        ].to_numpy(
            dtype=np.float64
        )
        > 0.0
    )

    success_mask = (
        folds[
            "multimodal_minus_price_nn"
        ].to_numpy(
            dtype=np.float64
        )
        < 0.0
    )

    harmful_summary = (
        group_summary(
            folds,
            harmful_mask,
        )
    )

    helpful_summary = (
        group_summary(
            folds,
            helpful_mask,
        )
    )

    failure_summary = (
        group_summary(
            folds,
            failure_mask,
        )
    )

    success_summary = (
        group_summary(
            folds,
            success_mask,
        )
    )

    diagnostic_columns = (
        "mean_price_flow_cosine",
        "median_price_flow_cosine",
        "p10_price_flow_cosine",
        "p05_price_flow_cosine",
        "negative_cosine_fraction",
        "strong_conflict_fraction",
        "mean_price_context_norm",
        "mean_flow_context_norm",
        "mean_flow_price_norm_ratio",
        "median_flow_price_norm_ratio",
        "early_mean_cosine",
        "late_mean_cosine",
        "late_minus_early_cosine",
    )

    correlation_rows = []

    for target_name in (
        "flow_harm",
        "multimodal_minus_price_nn",
    ):
        target = folds[
            target_name
        ].to_numpy(
            dtype=np.float64
        )

        for diagnostic in (
            diagnostic_columns
        ):
            values = folds[
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

    # Does interference increase with longer selected training?
    epochs = folds[
        "epochs"
    ].to_numpy(
        dtype=np.float64
    )

    training_relationships = {}

    for diagnostic in (
        "negative_cosine_fraction",
        "strong_conflict_fraction",
        "late_minus_early_cosine",
        "late_mean_cosine",
    ):
        values = folds[
            diagnostic
        ].to_numpy(
            dtype=np.float64
        )

        training_relationships[
            diagnostic
        ] = {
            "pearson":
                correlation(
                    epochs,
                    values,
                    method="pearson",
                ),
            "spearman":
                correlation(
                    epochs,
                    values,
                    method="spearman",
                ),
        }

    diagnostics = {
        "baseline_reproduction_error":
            reproduction_error,
        "aggregate": {
            "steps":
                int(
                    cosine.size
                ),
            "cosine":
                descriptive(
                    cosine
                ),
            "negative_fraction":
                float(
                    np.mean(
                        cosine < 0.0
                    )
                ),
            "strong_conflict_fraction":
                float(
                    np.mean(
                        cosine < -0.25
                    )
                ),
        },
        "training_progress":
            quartile_rows,
        "flow_harm_groups": {
            "harmful":
                harmful_summary,
            "helpful":
                helpful_summary,
        },
        "multimodal_performance_groups": {
            "worse_than_price":
                failure_summary,
            "better_than_price":
                success_summary,
        },
        "epochs_vs_interference":
            training_relationships,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            diagnostics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "GRADIENT-INTERFERENCE DIAGNOSTICS"
    )

    print(
        "================================="
    )

    print()
    print(
        "BASELINE REPRODUCTION"
    )

    print(
        "---------------------"
    )

    print(
        "Maximum absolute probability error:",
        f"{reproduction_error:.12g}",
    )

    print()
    print(
        "AGGREGATE CONFLICT"
    )

    print(
        "------------------"
    )

    print(
        "Steps:",
        cosine.size,
    )

    print(
        "Mean cosine:",
        f"{np.mean(cosine):+.4f}",
    )

    print(
        "Median cosine:",
        f"{np.median(cosine):+.4f}",
    )

    print(
        "Negative cosine fraction:",
        f"{np.mean(cosine < 0.0):.2%}",
    )

    print(
        "Strong conflict fraction:",
        f"{np.mean(cosine < -0.25):.2%}",
    )

    print()
    print(
        "CONFLICT ACROSS TRAINING"
    )

    print(
        "------------------------"
    )

    for row in quartile_rows:
        print(
            f"{row['quartile']} "
            f"meanCos="
            f"{row['mean_cosine']:+.3f} "
            f"medianCos="
            f"{row['median_cosine']:+.3f} "
            f"negative="
            f"{row['negative_fraction']:.1%} "
            f"strong="
            f"{row['strong_conflict_fraction']:.1%}"
        )

    print()
    print(
        "FLOW-HARMFUL VS FLOW-HELPFUL FOLDS"
    )

    print(
        "----------------------------------"
    )

    for name, summary in (
        (
            "Harmful",
            harmful_summary,
        ),
        (
            "Helpful",
            helpful_summary,
        ),
    ):
        print(
            f"{name:8s} "
            f"n={summary['folds']:2d} "
            f"meanCos="
            f"{summary['mean_price_flow_cosine']:+.3f} "
            f"negative="
            f"{summary['negative_cosine_fraction']:.1%} "
            f"strong="
            f"{summary['strong_conflict_fraction']:.1%} "
            f"lateCos="
            f"{summary['late_mean_cosine']:+.3f} "
            f"lateEarly="
            f"{summary['late_minus_early_cosine']:+.3f} "
            f"Flow/PriceNorm="
            f"{summary['mean_flow_price_norm_ratio']:.3f}"
        )

    print()
    print(
        "MULTIMODAL FAILURE VS SUCCESS FOLDS"
    )

    print(
        "-----------------------------------"
    )

    for name, summary in (
        (
            "Worse",
            failure_summary,
        ),
        (
            "Better",
            success_summary,
        ),
    ):
        print(
            f"{name:8s} "
            f"n={summary['folds']:2d} "
            f"meanCos="
            f"{summary['mean_price_flow_cosine']:+.3f} "
            f"negative="
            f"{summary['negative_cosine_fraction']:.1%} "
            f"strong="
            f"{summary['strong_conflict_fraction']:.1%} "
            f"lateCos="
            f"{summary['late_mean_cosine']:+.3f} "
            f"lateEarly="
            f"{summary['late_minus_early_cosine']:+.3f}"
        )

    for target_name, title in (
        (
            "flow_harm",
            "INTERFERENCE VS FLOW HARM",
        ),
        (
            "multimodal_minus_price_nn",
            "INTERFERENCE VS MULTIMODAL FAILURE",
        ),
    ):
        print()
        print(
            title
        )

        print(
            "-" * len(title)
        )

        subset = correlations[
            correlations[
                "target"
            ]
            == target_name
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
                f"{row['diagnostic']:32s} "
                f"Pearson="
                f"{row['pearson']:+.3f} "
                f"Spearman="
                f"{row['spearman']:+.3f}"
            )

    print()
    print(
        "SELECTED EPOCHS VS INTERFERENCE"
    )

    print(
        "-------------------------------"
    )

    for diagnostic, item in (
        training_relationships.items()
    ):
        print(
            f"{diagnostic:32s} "
            f"Pearson="
            f"{item['pearson']:+.3f} "
            f"Spearman="
            f"{item['spearman']:+.3f}"
        )

    print()
    print(
        "WORST FLOW-HARM FOLDS"
    )

    print(
        "---------------------"
    )

    worst = folds.sort_values(
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
            f"{row['multimodal_minus_price_nn']:+.6f} "
            f"epochs="
            f"{int(row['epochs']):2d} "
            f"meanCos="
            f"{row['mean_price_flow_cosine']:+.3f} "
            f"neg="
            f"{row['negative_cosine_fraction']:.1%} "
            f"strong="
            f"{row['strong_conflict_fraction']:.1%} "
            f"early="
            f"{row['early_mean_cosine']:+.3f} "
            f"late="
            f"{row['late_mean_cosine']:+.3f}"
        )

    print()
    print(
        "Diagnostics ->",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
