from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_synthetic_optimizer_stress_benchmark as stress


ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/synthetic_mom_adamw"
)

RUN_FILE = (
    ROOT
    / "synthetic_mom_adamw_runs.csv"
)

RESULTS_FILE = (
    ROOT
    / "synthetic_mom_adamw_benchmark.json"
)

OUTPUT_FILE = (
    ROOT
    / "synthetic_mom_adamw_diagnostics.json"
)


def paired(
    values: np.ndarray,
) -> dict[str, float | int]:
    return stress.paired_summary(
        np.asarray(
            values,
            dtype=np.float64,
        )
    )


def rows_for(
    frame: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    result = (
        frame[
            frame[
                "scenario"
            ]
            == scenario
        ]
        .sort_values(
            "seed"
        )
        .reset_index(
            drop=True
        )
    )

    if result.shape[0] != 12:
        raise RuntimeError(
            f"Expected 12 rows for {scenario}, "
            f"found {result.shape[0]}."
        )

    return result


def main() -> None:
    if not RUN_FILE.exists():
        raise FileNotFoundError(
            "Run the MoM-AdamW benchmark first."
        )

    frame = pd.read_csv(
        RUN_FILE
    )

    scenarios = [
        "stationary_gaussian",
        "stationary_heavy_tail",
        "flow_sign_shift_gaussian",
        "flow_sign_shift_heavy_tail",
        "flow_scale_shift",
    ]

    diagnostics = {}

    print(
        "MoM-ADAMW DIAGNOSTICS"
    )

    print(
        "====================="
    )

    print()
    print(
        "SCENARIO COMPARISONS"
    )

    print(
        "--------------------"
    )

    for scenario in scenarios:
        rows = rows_for(
            frame,
            scenario,
        )

        mom = rows[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )

        adamw = rows[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )

        price = rows[
            "price_log_loss"
        ].to_numpy(
            dtype=np.float64
        )

        delta_adamw = (
            mom
            - adamw
        )

        delta_price = (
            mom
            - price
        )

        vs_adamw = paired(
            delta_adamw
        )

        vs_price = paired(
            delta_price
        )

        diagnostics[
            scenario
        ] = {
            "mom_mean_log_loss":
                float(
                    np.mean(
                        mom
                    )
                ),
            "adamw_mean_log_loss":
                float(
                    np.mean(
                        adamw
                    )
                ),
            "price_mean_log_loss":
                float(
                    np.mean(
                        price
                    )
                ),
            "mom_vs_adamw":
                vs_adamw,
            "mom_vs_price":
                vs_price,
            "mean_selected_epoch":
                float(
                    rows[
                        "selected_epoch"
                    ].mean()
                ),
            "median_group_outlier_ratio":
                float(
                    rows[
                        "median_group_outlier_ratio"
                    ].median()
                ),
            "mean_p95_group_outlier_ratio":
                float(
                    rows[
                        "p95_group_outlier_ratio"
                    ].mean()
                ),
        }

        print()
        print(
            scenario
        )

        print(
            "-" * len(
                scenario
            )
        )

        print(
            "AdamW LL:",
            f"{np.mean(adamw):.6f}"
        )

        print(
            "MoM LL:",
            f"{np.mean(mom):.6f}"
        )

        print(
            "MoM - AdamW:",
            f"{vs_adamw['mean']:+.6f}"
        )

        print(
            "SE:",
            f"{vs_adamw['standard_error']:.6f}"
        )

        print(
            "t:",
            f"{vs_adamw['t_statistic']:+.3f}"
        )

        print(
            "p:",
            f"{vs_adamw['normal_p_value']:.6f}"
        )

        print(
            "MoM wins:",
            f"{vs_adamw['wins']}/12"
        )

        print(
            "MoM - Price:",
            f"{vs_price['mean']:+.6f}"
        )

        print(
            "Selected epoch:",
            f"{rows['selected_epoch'].mean():.2f}"
        )

        print(
            "Median block-outlier ratio:",
            f"{rows['median_group_outlier_ratio'].median():.3f}"
        )

    stationary = rows_for(
        frame,
        "stationary_gaussian",
    )

    heavy = rows_for(
        frame,
        "stationary_heavy_tail",
    )

    sign_gaussian = rows_for(
        frame,
        "flow_sign_shift_gaussian",
    )

    sign_heavy = rows_for(
        frame,
        "flow_sign_shift_heavy_tail",
    )

    scale_shift = rows_for(
        frame,
        "flow_scale_shift",
    )

    stationary_mom = (
        stationary[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    stationary_adamw = (
        stationary[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    heavy_mom = (
        heavy[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    heavy_adamw = (
        heavy[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    sign_gaussian_mom = (
        sign_gaussian[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    sign_gaussian_adamw = (
        sign_gaussian[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    sign_heavy_mom = (
        sign_heavy[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    sign_heavy_adamw = (
        sign_heavy[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    scale_mom = (
        scale_shift[
            "test_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    scale_adamw = (
        scale_shift[
            "adamw_log_loss"
        ].to_numpy(
            dtype=np.float64
        )
    )

    mom_heavy_penalty = (
        heavy_mom
        - stationary_mom
    )

    adamw_heavy_penalty = (
        heavy_adamw
        - stationary_adamw
    )

    heavy_penalty_reduction = (
        mom_heavy_penalty
        - adamw_heavy_penalty
    )

    mom_sign_penalty = (
        sign_gaussian_mom
        - stationary_mom
    )

    adamw_sign_penalty = (
        sign_gaussian_adamw
        - stationary_adamw
    )

    sign_penalty_change = (
        mom_sign_penalty
        - adamw_sign_penalty
    )

    mom_combined_penalty = (
        sign_heavy_mom
        - stationary_mom
    )

    adamw_combined_penalty = (
        sign_heavy_adamw
        - stationary_adamw
    )

    combined_penalty_reduction = (
        mom_combined_penalty
        - adamw_combined_penalty
    )

    mom_scale_penalty = (
        scale_mom
        - stationary_mom
    )

    adamw_scale_penalty = (
        scale_adamw
        - stationary_adamw
    )

    scale_penalty_change = (
        mom_scale_penalty
        - adamw_scale_penalty
    )

    print()
    print(
        "ROBUSTNESS DECOMPOSITION"
    )

    print(
        "------------------------"
    )

    print(
        "Stationary heavy-tail penalty"
    )

    print(
        "  AdamW:",
        f"{np.mean(adamw_heavy_penalty):+.6f}"
    )

    print(
        "  MoM:",
        f"{np.mean(mom_heavy_penalty):+.6f}"
    )

    print(
        "  MoM minus AdamW penalty:",
        f"{np.mean(heavy_penalty_reduction):+.6f}"
    )

    heavy_reduction_summary = paired(
        heavy_penalty_reduction
    )

    print(
        "  p:",
        f"{heavy_reduction_summary['normal_p_value']:.6f}"
    )

    print()
    print(
        "Gaussian sign-shift penalty"
    )

    print(
        "  AdamW:",
        f"{np.mean(adamw_sign_penalty):+.6f}"
    )

    print(
        "  MoM:",
        f"{np.mean(mom_sign_penalty):+.6f}"
    )

    print(
        "  MoM minus AdamW penalty:",
        f"{np.mean(sign_penalty_change):+.6f}"
    )

    print()
    print(
        "Heavy-tail + sign-shift penalty"
    )

    print(
        "  AdamW:",
        f"{np.mean(adamw_combined_penalty):+.6f}"
    )

    print(
        "  MoM:",
        f"{np.mean(mom_combined_penalty):+.6f}"
    )

    print(
        "  MoM minus AdamW penalty:",
        f"{np.mean(combined_penalty_reduction):+.6f}"
    )

    combined_summary = paired(
        combined_penalty_reduction
    )

    print(
        "  p:",
        f"{combined_summary['normal_p_value']:.6f}"
    )

    print()
    print(
        "Flow scale-shift penalty"
    )

    print(
        "  AdamW:",
        f"{np.mean(adamw_scale_penalty):+.6f}"
    )

    print(
        "  MoM:",
        f"{np.mean(mom_scale_penalty):+.6f}"
    )

    print(
        "  MoM minus AdamW penalty:",
        f"{np.mean(scale_penalty_change):+.6f}"
    )

    print()
    print(
        "GAUSSIAN PERFORMANCE PRESERVATION"
    )

    print(
        "---------------------------------"
    )

    stationary_delta = (
        stationary_mom
        - stationary_adamw
    )

    stationary_summary = paired(
        stationary_delta
    )

    print(
        "MoM - AdamW:",
        f"{stationary_summary['mean']:+.6f}"
    )

    print(
        "p:",
        f"{stationary_summary['normal_p_value']:.6f}"
    )

    print(
        "MoM wins:",
        f"{stationary_summary['wins']}/12"
    )

    output = {
        "experiment":
            "synthetic-mom-adamw-diagnostics",
        "scenario_diagnostics":
            diagnostics,
        "robustness_decomposition": {
            "stationary_heavy_tail": {
                "adamw_penalty":
                    float(
                        np.mean(
                            adamw_heavy_penalty
                        )
                    ),
                "mom_penalty":
                    float(
                        np.mean(
                            mom_heavy_penalty
                        )
                    ),
                "mom_minus_adamw_penalty":
                    heavy_reduction_summary,
            },
            "flow_sign_shift_gaussian": {
                "adamw_penalty":
                    float(
                        np.mean(
                            adamw_sign_penalty
                        )
                    ),
                "mom_penalty":
                    float(
                        np.mean(
                            mom_sign_penalty
                        )
                    ),
                "mom_minus_adamw_penalty":
                    paired(
                        sign_penalty_change
                    ),
            },
            "flow_sign_shift_heavy_tail": {
                "adamw_penalty":
                    float(
                        np.mean(
                            adamw_combined_penalty
                        )
                    ),
                "mom_penalty":
                    float(
                        np.mean(
                            mom_combined_penalty
                        )
                    ),
                "mom_minus_adamw_penalty":
                    combined_summary,
            },
            "flow_scale_shift": {
                "adamw_penalty":
                    float(
                        np.mean(
                            adamw_scale_penalty
                        )
                    ),
                "mom_penalty":
                    float(
                        np.mean(
                            mom_scale_penalty
                        )
                    ),
                "mom_minus_adamw_penalty":
                    paired(
                        scale_penalty_change
                    ),
            },
        },
        "stationary_gaussian_preservation":
            stationary_summary,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "DECISION CHECK"
    )

    print(
        "--------------"
    )

    heavy_improves = (
        heavy_reduction_summary[
            "mean"
        ]
        < 0.0
    )

    combined_improves = (
        combined_summary[
            "mean"
        ]
        < 0.0
    )

    gaussian_damage = (
        stationary_summary[
            "mean"
        ]
    )

    heavy_significant = (
        heavy_reduction_summary[
            "normal_p_value"
        ]
        < 0.05
    )

    combined_significant = (
        combined_summary[
            "normal_p_value"
        ]
        < 0.05
    )

    if (
        heavy_improves
        and heavy_significant
        and gaussian_damage <= 0.002
    ):
        print(
            "MoM provides evidence of useful "
            "stationary heavy-tail robustness."
        )

    elif (
        combined_improves
        and combined_significant
        and gaussian_damage <= 0.002
    ):
        print(
            "MoM benefit appears specific to "
            "heavy-tail/non-stationarity interaction."
        )

    else:
        print(
            "MoM does not provide a compelling "
            "robust-gradient advantage."
        )

    print()
    print(
        "Diagnostics ->",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
