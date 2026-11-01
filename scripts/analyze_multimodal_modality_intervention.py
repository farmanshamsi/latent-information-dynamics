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
    "latent_neural/modality_intervention"
)

RESULTS_FILE = (
    ROOT
    / "multimodal_modality_intervention.json"
)

FOLD_FILE = (
    ROOT
    / "multimodal_modality_intervention_folds.csv"
)

DRIFT_FILE = Path(
    "data/derived/experiments/"
    "latent_neural/regime_drift/"
    "multimodal_regime_drift_folds.csv"
)

OUTPUT_FILE = (
    ROOT
    / "multimodal_modality_intervention_diagnostics.json"
)

CORRELATION_FILE = (
    ROOT
    / "multimodal_modality_intervention_diagnostics_correlations.csv"
)


def safe_repair_fraction(
    baseline_gap: np.ndarray,
    repair: np.ndarray,
) -> np.ndarray:
    result = np.full(
        baseline_gap.shape,
        np.nan,
        dtype=np.float64,
    )

    mask = (
        baseline_gap > 1e-12
    )

    result[
        mask
    ] = (
        repair[
            mask
        ]
        / baseline_gap[
            mask
        ]
    )

    return result


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Run the modality-intervention "
            "experiment first."
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

    full_gap = (
        folds[
            "full_minus_price_nn"
        ].to_numpy(
            dtype=np.float64
        )
    )

    flow_delta = (
        folds[
            "flow_neutral_minus_full"
        ].to_numpy(
            dtype=np.float64
        )
    )

    price_delta = (
        folds[
            "price_neutral_minus_full"
        ].to_numpy(
            dtype=np.float64
        )
    )

    flow_repair = (
        -flow_delta
    )

    price_damage = (
        price_delta
    )

    repair_fraction = (
        safe_repair_fraction(
            full_gap,
            flow_repair,
        )
    )

    flow_helpful_folds = int(
        np.sum(
            flow_delta > 0.0
        )
    )

    flow_harmful_folds = int(
        np.sum(
            flow_delta < 0.0
        )
    )

    flow_neutral_folds = (
        20
        - flow_helpful_folds
        - flow_harmful_folds
    )

    positive_gap_mask = (
        full_gap > 0.0
    )

    failing_fold_count = int(
        np.sum(
            positive_gap_mask
        )
    )

    failing_fold_repair_count = int(
        np.sum(
            (
                flow_repair > 0.0
            )
            & positive_gap_mask
        )
    )

    correlations = []

    if DRIFT_FILE.exists():
        drift = pd.read_csv(
            DRIFT_FILE
        )

        merged = folds.merge(
            drift,
            on="fold",
            how="inner",
            validate="one_to_one",
        )

        drift_columns = [
            column
            for column
            in merged.columns
            if (
                column.startswith(
                    "flow_mean_shift"
                )
                or column.startswith(
                    "flow_scale_shift"
                )
                or column
                == "flow_covariance_shift"
            )
        ]

        repair = (
            -merged[
                "flow_neutral_minus_full"
            ].to_numpy(
                dtype=np.float64
            )
        )

        contribution = (
            merged[
                "price_neutral_minus_full"
            ].to_numpy(
                dtype=np.float64
            )
        )

        for column in drift_columns:
            values = merged[
                column
            ].to_numpy(
                dtype=np.float64
            )

            correlations.append(
                {
                    "diagnostic":
                        column,
                    "target":
                        "flow_neutralization_repair",
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

            correlations.append(
                {
                    "diagnostic":
                        column,
                    "target":
                        "price_neutralization_damage",
                    "pearson":
                        correlation(
                            values,
                            contribution,
                            method="pearson",
                        ),
                    "spearman":
                        correlation(
                            values,
                            contribution,
                            method="spearman",
                        ),
                }
            )

    diagnostics = {
        "baseline_reproduction_error":
            reproduction_error,
        "flow_inference_effect": {
            "folds_where_flow_helps":
                flow_helpful_folds,
            "folds_where_flow_hurts":
                flow_harmful_folds,
            "folds_effectively_tied":
                flow_neutral_folds,
        },
        "bad_multimodal_folds": {
            "count":
                failing_fold_count,
            "repaired_by_flow_neutralization":
                failing_fold_repair_count,
        },
        "mean_flow_neutralization_repair":
            float(
                np.mean(
                    flow_repair
                )
            ),
        "median_flow_neutralization_repair":
            float(
                np.median(
                    flow_repair
                )
            ),
        "mean_price_neutralization_damage":
            float(
                np.mean(
                    price_damage
                )
            ),
        "positive_gap_repair_fraction": {
            "mean":
                float(
                    np.nanmean(
                        repair_fraction
                    )
                ),
            "median":
                float(
                    np.nanmedian(
                        repair_fraction
                    )
                ),
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            diagnostics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if correlations:
        pd.DataFrame(
            correlations
        ).to_csv(
            CORRELATION_FILE,
            index=False,
        )

    print(
        "MULTIMODAL MODALITY-INTERVENTION DIAGNOSTICS"
    )

    print(
        "==========================================="
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

    print()
    print(
        "FLOW INFERENCE CONTRIBUTION"
    )

    print(
        "---------------------------"
    )

    print(
        "Flow helps full model in:",
        f"{flow_helpful_folds}/20 folds",
    )

    print(
        "Flow hurts full model in:",
        f"{flow_harmful_folds}/20 folds",
    )

    print(
        "Mean Flow-neutralization repair:",
        f"{np.mean(flow_repair):+.6f}",
    )

    print(
        "Median Flow-neutralization repair:",
        f"{np.median(flow_repair):+.6f}",
    )

    print()
    print(
        "FAILURE-FOLD REPAIR"
    )

    print(
        "-------------------"
    )

    print(
        "Multimodal-worse-than-Price folds:",
        failing_fold_count,
    )

    print(
        "Those improved by Flow neutralization:",
        failing_fold_repair_count,
    )

    print(
        "Mean repair fraction on bad folds:",
        f"{np.nanmean(repair_fraction):+.3f}",
    )

    print(
        "Median repair fraction on bad folds:",
        f"{np.nanmedian(repair_fraction):+.3f}",
    )

    print()
    print(
        "WORST MULTIMODAL FOLDS"
    )

    print(
        "----------------------"
    )

    ranked = folds.copy()

    ranked[
        "flow_repair"
    ] = (
        -ranked[
            "flow_neutral_minus_full"
        ]
    )

    ranked[
        "repair_fraction"
    ] = (
        safe_repair_fraction(
            ranked[
                "full_minus_price_nn"
            ].to_numpy(
                dtype=np.float64
            ),
            ranked[
                "flow_repair"
            ].to_numpy(
                dtype=np.float64
            ),
        )
    )

    ranked = ranked.sort_values(
        "full_minus_price_nn",
        ascending=False,
    )

    for _, row in (
        ranked.head(
            10
        ).iterrows()
    ):
        fraction = row[
            "repair_fraction"
        ]

        fraction_text = (
            "NA"
            if not np.isfinite(
                fraction
            )
            else f"{fraction:+.2f}"
        )

        print(
            f"Fold {int(row['fold']):02d} "
            f"gap="
            f"{row['full_minus_price_nn']:+.6f} "
            f"Flow0-repair="
            f"{row['flow_repair']:+.6f} "
            f"repair-frac="
            f"{fraction_text} "
            f"Flow0-vs-Price="
            f"{row['flow_neutral_minus_price_nn']:+.6f}"
        )

    print()
    print(
        "PRICE REMOVAL EFFECT"
    )

    print(
        "--------------------"
    )

    print(
        "Mean Price-neutral minus Full:",
        f"{np.mean(price_damage):+.6f}",
    )

    print(
        "Median Price-neutral minus Full:",
        f"{np.median(price_damage):+.6f}",
    )

    if correlations:
        correlation_frame = pd.DataFrame(
            correlations
        )

        print()
        print(
            "FLOW DRIFT VS FLOW-NEUTRALIZATION REPAIR"
        )

        print(
            "----------------------------------------"
        )

        subset = correlation_frame[
            correlation_frame[
                "target"
            ]
            == "flow_neutralization_repair"
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
            subset.iterrows()
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
        "MECHANISM CHECK"
    )

    print(
        "---------------"
    )

    flow_vs_full = (
        results[
            "flow_neutral_vs_full"
        ][
            "log_loss"
        ]
    )

    flow_vs_price = (
        results[
            "flow_neutral_vs_price_nn"
        ][
            "log_loss"
        ]
    )

    price_vs_full = (
        results[
            "price_neutral_vs_full"
        ][
            "log_loss"
        ]
    )

    print(
        "Flow-neutral minus Full LL:",
        f"{flow_vs_full['mean_difference']:+.8f}",
    )

    print(
        "Flow-neutral minus Price NN LL:",
        f"{flow_vs_price['mean_difference']:+.8f}",
    )

    print(
        "Price-neutral minus Full LL:",
        f"{price_vs_full['mean_difference']:+.8f}",
    )

    print()
    print(
        "Flow-neutral vs Full HAC p-values:"
    )

    for lag, item in (
        flow_vs_full[
            "hac"
        ].items()
    ):
        print(
            f"  lag {int(lag):2d}: "
            f"t={item['t_statistic']:+.4f} "
            f"p={item['p_value']:.6f}"
        )

    print()
    print(
        "Diagnostics ->",
        OUTPUT_FILE,
    )

    if correlations:
        print(
            "Correlations ->",
            CORRELATION_FILE,
        )


if __name__ == "__main__":
    main()
