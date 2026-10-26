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
    "latent_neural/optimizer_clipping/"
    "matched_epoch_control"
)

RESULTS_FILE = (
    ROOT
    / "matched_epoch_clipping_control.json"
)

FOLD_FILE = (
    ROOT
    / "matched_epoch_clipping_folds.csv"
)

OUTPUT_FILE = (
    ROOT
    / "matched_epoch_clipping_diagnostics.json"
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


def main() -> None:
    results = json.loads(
        RESULTS_FILE.read_text(
            encoding="utf-8"
        )
    )

    folds = pd.read_csv(
        FOLD_FILE
    )

    if folds.shape[0] != 20:
        raise RuntimeError(
            "Expected 20 outer folds."
        )

    reproduction_error = float(
        results[
            "baseline_reproduction"
        ][
            "maximum_absolute_probability_error"
        ]
    )

    print(
        "MATCHED-EPOCH CLIPPING DIAGNOSTICS"
    )

    print(
        "=================================="
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
        "AGGREGATE METRICS"
    )

    print(
        "-----------------"
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

    diagnostics = {
        "reproduction_error":
            reproduction_error,
        "thresholds": {},
    }

    print()
    print(
        "MATCHED CLIPPING VS ADAMW"
    )

    print(
        "-------------------------"
    )

    for threshold in THRESHOLDS:
        key = threshold_key(
            threshold
        )

        comparison = (
            results[
                "clipping_vs_adamw"
            ][
                key
            ]
        )

        price_comparison = (
            results[
                "clipping_vs_price_nn"
            ][
                key
            ]
        )

        gradient = (
            results[
                "gradient_control"
            ][
                key
            ]
        )

        fold_delta_column = (
            f"{key}_minus_adamw"
        )

        deltas = pd.to_numeric(
            folds[
                fold_delta_column
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        wins = int(
            np.sum(
                deltas < 0.0
            )
        )

        losses = int(
            np.sum(
                deltas > 0.0
            )
        )

        ties = (
            20
            - wins
            - losses
        )

        ll = comparison[
            "log_loss"
        ]

        brier = comparison[
            "brier"
        ]

        ll_price = (
            price_comparison[
                "log_loss"
            ]
        )

        print()
        print(
            f"clip={threshold:.2f}"
        )

        print(
            f"  vs AdamW LL:    "
            f"{ll['mean_difference']:+.8f}"
        )

        print(
            f"  vs AdamW Brier: "
            f"{brier['mean_difference']:+.8f}"
        )

        print(
            f"  vs Price NN LL: "
            f"{ll_price['mean_difference']:+.8f}"
        )

        print(
            "  Fold wins/losses/ties:",
            f"{wins}/{losses}/{ties}",
        )

        print(
            "  Clip rate:",
            f"{gradient['clipped_fraction']:.2%}",
        )

        print(
            "  Raw gradient:",
            f"p95={gradient['raw_p95']:.6f}",
            f"p99={gradient['raw_p99']:.6f}",
            f"max={gradient['raw_max']:.6f}",
        )

        print(
            "  Effective gradient:",
            f"p95={gradient['effective_p95']:.6f}",
            f"p99={gradient['effective_p99']:.6f}",
            f"max={gradient['effective_max']:.6f}",
        )

        print(
            "  HAC LL:"
        )

        for lag in (
            0,
            1,
            6,
            12,
            42,
        ):
            item = ll[
                "hac"
            ][
                str(
                    lag
                )
            ]

            print(
                f"    lag {lag:2d}: "
                f"t="
                f"{item['t_statistic']:+.4f} "
                f"p="
                f"{item['p_value']:.6f}"
            )

        diagnostics[
            "thresholds"
        ][
            key
        ] = {
            "ll_delta_vs_adamw":
                ll[
                    "mean_difference"
                ],
            "brier_delta_vs_adamw":
                brier[
                    "mean_difference"
                ],
            "ll_delta_vs_price_nn":
                ll_price[
                    "mean_difference"
                ],
            "fold_wins":
                wins,
            "fold_losses":
                losses,
            "fold_ties":
                ties,
            "clipped_fraction":
                gradient[
                    "clipped_fraction"
                ],
        }

    print()
    print(
        "FAILURE-REGIME REPAIR"
    )

    print(
        "---------------------"
    )

    folds = folds.copy()

    folds[
        "baseline_gap"
    ] = (
        folds[
            "adamw_log_loss"
        ]
        - folds[
            "price_nn_log_loss"
        ]
    )

    worst = (
        folds
        .sort_values(
            "baseline_gap",
            ascending=False,
        )
        .head(
            7
        )
    )

    for _, row in worst.iterrows():
        text = (
            f"Fold {int(row['fold']):02d} "
            f"baseline-gap="
            f"{row['baseline_gap']:+.6f}"
        )

        for threshold in THRESHOLDS:
            key = threshold_key(
                threshold
            )

            improvement = (
                -float(
                    row[
                        f"{key}_minus_adamw"
                    ]
                )
            )

            text += (
                f" C{threshold:.2f}-repair="
                f"{improvement:+.6f}"
            )

        print(
            text
        )

    print()
    print(
        "FAILURE SEVERITY / REPAIR CORRELATIONS"
    )

    print(
        "--------------------------------------"
    )

    baseline_gap = (
        folds[
            "baseline_gap"
        ].to_numpy(
            dtype=np.float64
        )
    )

    repair_correlations = {}

    for threshold in THRESHOLDS:
        key = threshold_key(
            threshold
        )

        repair = (
            -folds[
                f"{key}_minus_adamw"
            ].to_numpy(
                dtype=np.float64
            )
        )

        pearson = correlation(
            baseline_gap,
            repair,
            method="pearson",
        )

        spearman = correlation(
            baseline_gap,
            repair,
            method="spearman",
        )

        repair_correlations[
            key
        ] = {
            "pearson":
                pearson,
            "spearman":
                spearman,
        }

        print(
            f"clip={threshold:.2f} "
            f"Pearson="
            f"{pearson:+.3f} "
            f"Spearman="
            f"{spearman:+.3f}"
        )

    diagnostics[
        "failure_repair_correlations"
    ] = repair_correlations

    print()
    print(
        "EPOCH / CLIPPING EFFECT"
    )

    print(
        "-----------------------"
    )

    epochs = (
        folds[
            "epochs"
        ].to_numpy(
            dtype=np.float64
        )
    )

    for threshold in THRESHOLDS:
        key = threshold_key(
            threshold
        )

        improvement = (
            -folds[
                f"{key}_minus_adamw"
            ].to_numpy(
                dtype=np.float64
            )
        )

        pearson = correlation(
            epochs,
            improvement,
            method="pearson",
        )

        spearman = correlation(
            epochs,
            improvement,
            method="spearman",
        )

        print(
            f"clip={threshold:.2f} "
            f"epochs vs repair: "
            f"Pearson="
            f"{pearson:+.3f} "
            f"Spearman="
            f"{spearman:+.3f}"
        )

    OUTPUT_FILE.write_text(
        json.dumps(
            diagnostics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Diagnostics ->",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
