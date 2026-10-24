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
    "latent_neural/optimizer_clipping"
)

RESULTS_FILE = (
    ROOT
    / "nested_adamw_clipping_4h_2020_2025.json"
)

FOLD_FILE = (
    ROOT
    / "nested_adamw_clipping_folds.csv"
)

OUTPUT_FILE = (
    ROOT
    / "optimizer_clipping_diagnostics.json"
)

CORRELATION_FILE = (
    ROOT
    / "optimizer_clipping_fold_correlations.csv"
)


THRESHOLDS = (
    0.10,
    0.20,
    0.40,
)


def threshold_key(
    threshold: float,
) -> str:
    return (
        f"clip_{threshold:.2f}"
        .replace(
            ".",
            "_",
        )
    )


def finite_array(
    series: pd.Series,
) -> np.ndarray:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    return values[
        np.isfinite(
            values
        )
    ]


def fold_summary(
    folds: pd.DataFrame,
) -> dict[str, object]:
    adaptive_delta = finite_array(
        folds[
            "adaptive_minus_adamw"
        ]
    )

    baseline_gap = (
        pd.to_numeric(
            folds[
                "adamw_log_loss"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            folds[
                "price_nn_log_loss"
            ],
            errors="coerce",
        )
    ).to_numpy(
        dtype=np.float64
    )

    adaptive_repair = (
        -pd.to_numeric(
            folds[
                "adaptive_minus_adamw"
            ],
            errors="coerce",
        )
    ).to_numpy(
        dtype=np.float64
    )

    result = {
        "adaptive_fold_wins":
            int(
                np.sum(
                    adaptive_delta
                    < 0.0
                )
            ),
        "adaptive_fold_losses":
            int(
                np.sum(
                    adaptive_delta
                    > 0.0
                )
            ),
        "adaptive_mean_fold_delta":
            float(
                np.mean(
                    adaptive_delta
                )
            ),
        "adaptive_median_fold_delta":
            float(
                np.median(
                    adaptive_delta
                )
            ),
        "adaptive_best_fold_delta":
            float(
                np.min(
                    adaptive_delta
                )
            ),
        "adaptive_worst_fold_delta":
            float(
                np.max(
                    adaptive_delta
                )
            ),
        "baseline_gap_vs_repair_pearson":
            correlation(
                baseline_gap,
                adaptive_repair,
                method="pearson",
            ),
        "baseline_gap_vs_repair_spearman":
            correlation(
                baseline_gap,
                adaptive_repair,
                method="spearman",
            ),
    }

    return result


def build_correlations(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    improvement = (
        -pd.to_numeric(
            folds[
                "adaptive_minus_adamw"
            ],
            errors="coerce",
        )
    ).to_numpy(
        dtype=np.float64
    )

    candidate_columns = [
        "adaptive_clipped_fraction",
        "adaptive_threshold",
        "adaptive_epochs",
        "adamw_log_loss",
        "price_nn_log_loss",
    ]

    for threshold in THRESHOLDS:
        key = threshold_key(
            threshold
        )

        candidate_columns.extend(
            [
                f"{key}_epochs",
                f"{key}_validation_log_loss",
                f"{key}_outer_log_loss",
                f"{key}_clipped_fraction",
            ]
        )

    rows = []

    for column in candidate_columns:
        values = pd.to_numeric(
            folds[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "diagnostic":
                    column,
                "pearson_with_clipping_improvement":
                    correlation(
                        values,
                        improvement,
                        method="pearson",
                    ),
                "spearman_with_clipping_improvement":
                    correlation(
                        values,
                        improvement,
                        method="spearman",
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Run the clipping experiment first: "
            f"{RESULTS_FILE}"
        )

    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"Missing fold file: {FOLD_FILE}"
        )

    results = json.loads(
        RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    folds = pd.read_csv(
        FOLD_FILE
    )

    if folds.shape[
        0
    ] != 20:
        raise RuntimeError(
            "Expected 20 outer folds."
        )

    correlations = (
        build_correlations(
            folds
        )
    )

    summary = {
        "experiment":
            "optimizer-clipping-diagnostics",
        "aggregate_metrics":
            results[
                "metrics"
            ],
        "adaptive_threshold_counts":
            results[
                "adaptive_threshold_counts"
            ],
        "adaptive_fold_summary":
            fold_summary(
                folds
            ),
        "fixed_threshold_vs_adamw":
            results[
                "fixed_threshold_vs_adamw"
            ],
        "adaptive_vs_adamw":
            results[
                "adaptive_vs_adamw"
            ],
        "adaptive_vs_price_nn":
            results[
                "adaptive_vs_price_nn"
            ],
        "interpretation_constraints": {
            "thresholds":
                (
                    "Thresholds were chosen after inspecting "
                    "2021-2025 gradient diagnostics. "
                    "This remains a developmental experiment."
                ),
            "correlations":
                (
                    "Fold-level correlations use only 20 "
                    "observations and are exploratory."
                ),
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    correlations.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    print(
        "OPTIMIZER CLIPPING DIAGNOSTICS"
    )

    print(
        "=============================="
    )

    print()
    print(
        "AGGREGATE"
    )

    print(
        "---------"
    )

    for name, metrics in (
        results[
            "metrics"
        ].items()
    ):
        print(
            f"{name:28s} "
            f"LL="
            f"{metrics['log_loss']:.8f} "
            f"Brier="
            f"{metrics['brier_score']:.8f} "
            f"Acc="
            f"{metrics['accuracy']:.4%}"
        )

    print()
    print(
        "FIXED CLIPPING VS ADAMW"
    )

    print(
        "-----------------------"
    )

    for threshold in THRESHOLDS:
        key = threshold_key(
            threshold
        )

        comparison = (
            results[
                "fixed_threshold_vs_adamw"
            ][
                key
            ]
        )

        ll = comparison[
            "log_loss"
        ]

        brier = comparison[
            "brier"
        ]

        print(
            f"clip={threshold:.2f} "
            f"LL delta="
            f"{ll['mean_difference']:+.8f} "
            f"Brier delta="
            f"{brier['mean_difference']:+.8f}"
        )

        print(
            "   HAC LL p-values:",
            " ".join(
                (
                    f"{lag}:"
                    f"{ll['hac'][str(lag)]['p_value']:.4f}"
                )
                for lag
                in (
                    0,
                    1,
                    6,
                    12,
                    42,
                )
            ),
        )

    print()
    print(
        "ADAPTIVE CLIPPING VS ADAMW"
    )

    print(
        "--------------------------"
    )

    adaptive = (
        results[
            "adaptive_vs_adamw"
        ]
    )

    print(
        "LL delta:",
        f"{adaptive['log_loss']['mean_difference']:+.8f}",
    )

    print(
        "Brier delta:",
        f"{adaptive['brier']['mean_difference']:+.8f}",
    )

    for lag in (
        0,
        1,
        6,
        12,
        42,
    ):
        item = adaptive[
            "log_loss"
        ][
            "hac"
        ][
            str(
                lag
            )
        ]

        print(
            f"lag {lag:2d}: "
            f"t="
            f"{item['t_statistic']:+.4f} "
            f"p="
            f"{item['p_value']:.6f}"
        )

    print()
    print(
        "ADAPTIVE CLIPPING VS PRICE NN"
    )

    print(
        "-----------------------------"
    )

    adaptive_price = (
        results[
            "adaptive_vs_price_nn"
        ]
    )

    print(
        "LL delta:",
        f"{adaptive_price['log_loss']['mean_difference']:+.8f}",
    )

    print(
        "Brier delta:",
        f"{adaptive_price['brier']['mean_difference']:+.8f}",
    )

    print()
    print(
        "THRESHOLD SELECTION"
    )

    print(
        "-------------------"
    )

    print(
        results[
            "adaptive_threshold_counts"
        ]
    )

    summary_folds = (
        fold_summary(
            folds
        )
    )

    print(
        "Adaptive fold wins:",
        summary_folds[
            "adaptive_fold_wins"
        ],
        "/20",
    )

    print(
        "Mean fold LL delta:",
        f"{summary_folds['adaptive_mean_fold_delta']:+.6f}",
    )

    print(
        "Median fold LL delta:",
        f"{summary_folds['adaptive_median_fold_delta']:+.6f}",
    )

    print()
    print(
        "FAILURE-REGIME REPAIR"
    )

    print(
        "---------------------"
    )

    folds = folds.copy()

    folds[
        "baseline_multimodal_gap"
    ] = (
        folds[
            "adamw_log_loss"
        ]
        - folds[
            "price_nn_log_loss"
        ]
    )

    folds[
        "clipping_improvement"
    ] = (
        -folds[
            "adaptive_minus_adamw"
        ]
    )

    worst = (
        folds
        .sort_values(
            "baseline_multimodal_gap",
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
            f"baseline-gap="
            f"{row['baseline_multimodal_gap']:+.6f} "
            f"clip-improvement="
            f"{row['clipping_improvement']:+.6f} "
            f"threshold="
            f"{row['adaptive_threshold']:.2f} "
            f"clip-rate="
            f"{row['adaptive_clipped_fraction']:.1%}"
        )

    print()
    print(
        "Baseline failure vs clipping repair:"
    )

    print(
        "Pearson =",
        f"{summary_folds['baseline_gap_vs_repair_pearson']:+.3f}",
    )

    print(
        "Spearman =",
        f"{summary_folds['baseline_gap_vs_repair_spearman']:+.3f}",
    )

    print()
    print(
        "TOP EXPLORATORY CORRELATIONS"
    )

    print(
        "----------------------------"
    )

    ranked = correlations.copy()

    ranked[
        "absolute_spearman"
    ] = (
        ranked[
            "spearman_with_clipping_improvement"
        ].abs()
    )

    ranked = ranked.sort_values(
        "absolute_spearman",
        ascending=False,
    )

    for _, row in (
        ranked.head(
            8
        ).iterrows()
    ):
        print(
            f"{row['diagnostic']:32s} "
            f"Pearson="
            f"{row['pearson_with_clipping_improvement']:+.3f} "
            f"Spearman="
            f"{row['spearman_with_clipping_improvement']:+.3f}"
        )

    print()
    print(
        "Diagnostics ->",
        OUTPUT_FILE,
    )

    print(
        "Correlations ->",
        CORRELATION_FILE,
    )


if __name__ == "__main__":
    main()
