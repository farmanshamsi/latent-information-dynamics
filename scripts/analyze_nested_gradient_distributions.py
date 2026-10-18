from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
    distribution_summary,
)


EXPERIMENT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/nested_walk_forward"
)

GRADIENT_FILE = (
    EXPERIMENT_ROOT
    / "nested_latent_refit_gradients.csv"
)

FOLD_FILE = (
    EXPERIMENT_ROOT
    / "nested_latent_folds.csv"
)

OUTPUT_ROOT = Path(
    "data/derived/experiments/"
    "latent_neural/gradient_diagnostics"
)

RESULTS_FILE = (
    OUTPUT_ROOT
    / "nested_gradient_distribution_diagnostics.json"
)

FOLD_OUTPUT_FILE = (
    OUTPUT_ROOT
    / "nested_gradient_fold_diagnostics.csv"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "nested_gradient_performance_correlations.csv"
)


GRADIENT_COLUMNS = (
    "total_norm",
    "price_encoder_norm",
    "latent_adapter_norm",
    "flow_encoder_norm",
    "fusion_norm",
    "prediction_head_norm",
)


def summarize_models(
    gradients: pd.DataFrame,
) -> dict[str, object]:
    result: dict[
        str,
        object,
    ] = {}

    for model_name, frame in (
        gradients
        .groupby(
            "model",
            sort=True,
        )
    ):
        model_result: dict[
            str,
            object,
        ] = {}

        for column in GRADIENT_COLUMNS:
            values = (
                pd.to_numeric(
                    frame[
                        column
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(
                    dtype=np.float64
                )
            )

            if values.size == 0:
                continue

            model_result[
                column
            ] = distribution_summary(
                values
            )

        if (
            model_name
            == "MULTIMODAL_NN"
        ):
            price = pd.to_numeric(
                frame[
                    "price_encoder_norm"
                ],
                errors="coerce",
            )

            flow = pd.to_numeric(
                frame[
                    "flow_encoder_norm"
                ],
                errors="coerce",
            )

            valid = (
                price.notna()
                & flow.notna()
                & (
                    price
                    > 0.0
                )
            )

            ratio = (
                flow[
                    valid
                ]
                / price[
                    valid
                ]
            ).to_numpy(
                dtype=np.float64
            )

            model_result[
                "flow_to_price_ratio"
            ] = distribution_summary(
                ratio
            )

        result[
            str(
                model_name
            )
        ] = model_result

    return result


def build_fold_diagnostics(
    gradients: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    multimodal = gradients[
        gradients[
            "model"
        ]
        == "MULTIMODAL_NN"
    ].copy()

    numeric_columns = (
        "loss",
        "total_norm",
        "price_encoder_norm",
        "flow_encoder_norm",
        "fusion_norm",
        "prediction_head_norm",
    )

    for column in numeric_columns:
        multimodal[
            column
        ] = pd.to_numeric(
            multimodal[
                column
            ],
            errors="coerce",
        )

    multimodal[
        "flow_to_price_ratio"
    ] = (
        multimodal[
            "flow_encoder_norm"
        ]
        / multimodal[
            "price_encoder_norm"
        ].replace(
            0.0,
            np.nan,
        )
    )

    records = []

    for fold, frame in (
        multimodal
        .groupby(
            "fold",
            sort=True,
        )
    ):
        total = (
            frame[
                "total_norm"
            ]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        ratio = (
            frame[
                "flow_to_price_ratio"
            ]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        total_summary = (
            distribution_summary(
                total
            )
        )

        ratio_summary = (
            distribution_summary(
                ratio
            )
        )

        records.append(
            {
                "fold":
                    int(
                        fold
                    ),
                "gradient_steps":
                    int(
                        frame.shape[
                            0
                        ]
                    ),
                "mean_training_loss":
                    float(
                        frame[
                            "loss"
                        ].mean()
                    ),
                "mean_total_norm":
                    total_summary[
                        "mean"
                    ],
                "p95_total_norm":
                    total_summary[
                        "p95"
                    ],
                "p99_total_norm":
                    total_summary[
                        "p99"
                    ],
                "max_total_norm":
                    total_summary[
                        "maximum"
                    ],
                "total_excess_kurtosis":
                    total_summary[
                        "excess_kurtosis"
                    ],
                "total_hill_alpha_05":
                    total_summary[
                        "hill_alpha_05"
                    ],
                "mean_price_norm":
                    float(
                        frame[
                            "price_encoder_norm"
                        ].mean()
                    ),
                "mean_flow_norm":
                    float(
                        frame[
                            "flow_encoder_norm"
                        ].mean()
                    ),
                "mean_flow_to_price_ratio":
                    ratio_summary[
                        "mean"
                    ],
                "p95_flow_to_price_ratio":
                    ratio_summary[
                        "p95"
                    ],
                "p99_flow_to_price_ratio":
                    ratio_summary[
                        "p99"
                    ],
                "max_flow_to_price_ratio":
                    ratio_summary[
                        "maximum"
                    ],
            }
        )

    diagnostics = pd.DataFrame(
        records
    )

    folds = folds.copy()

    folds[
        "fold"
    ] = pd.to_numeric(
        folds[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    return diagnostics.merge(
        folds,
        on="fold",
        how="left",
        validate="one_to_one",
    )


def build_correlations(
    fold_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    target = (
        fold_diagnostics[
            "multimodal_minus_price_nn"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    diagnostic_columns = (
        "mean_training_loss",
        "mean_total_norm",
        "p95_total_norm",
        "p99_total_norm",
        "max_total_norm",
        "total_excess_kurtosis",
        "total_hill_alpha_05",
        "mean_price_norm",
        "mean_flow_norm",
        "mean_flow_to_price_ratio",
        "p95_flow_to_price_ratio",
        "p99_flow_to_price_ratio",
        "max_flow_to_price_ratio",
    )

    records = []

    for column in diagnostic_columns:
        values = (
            fold_diagnostics[
                column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        records.append(
            {
                "diagnostic":
                    column,
                "pearson_with_oos_ll_delta":
                    correlation(
                        values,
                        target,
                        method="pearson",
                    ),
                "spearman_with_oos_ll_delta":
                    correlation(
                        values,
                        target,
                        method="spearman",
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def main() -> None:
    gradients = pd.read_csv(
        GRADIENT_FILE
    )

    folds = pd.read_csv(
        FOLD_FILE
    )

    model_summaries = (
        summarize_models(
            gradients
        )
    )

    fold_diagnostics = (
        build_fold_diagnostics(
            gradients,
            folds,
        )
    )

    correlations = (
        build_correlations(
            fold_diagnostics
        )
    )

    results = {
        "experiment":
            "nested-gradient-distribution-diagnostics",
        "interpretation_constraints": {
            "heavy_tail_claim":
                (
                    "Descriptive only. Hill estimates and "
                    "kurtosis do not establish infinite "
                    "variance or a finite-p moment model."
                ),
            "performance_correlation":
                (
                    "Exploratory across only 20 outer folds; "
                    "not causal evidence."
                ),
        },
        "model_gradient_distributions":
            model_summaries,
        "multimodal_fold_count":
            int(
                fold_diagnostics.shape[
                    0
                ]
            ),
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fold_diagnostics.to_csv(
        FOLD_OUTPUT_FILE,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    print(
        "GRADIENT DISTRIBUTION DIAGNOSTICS"
    )

    print(
        "================================="
    )

    for model, summaries in (
        model_summaries.items()
    ):
        print()
        print(
            model
        )

        print(
            "-" * len(
                model
            )
        )

        for gradient_name in (
            "total_norm",
            "price_encoder_norm",
            "flow_encoder_norm",
            "flow_to_price_ratio",
        ):
            if (
                gradient_name
                not in summaries
            ):
                continue

            item = summaries[
                gradient_name
            ]

            print(
                f"{gradient_name:24s} "
                f"n={item['count']:6d} "
                f"median={item['median']:.6f} "
                f"p95={item['p95']:.6f} "
                f"p99={item['p99']:.6f} "
                f"max={item['maximum']:.6f} "
                f"kurt={item['excess_kurtosis']:.3f} "
                f"hill5={item['hill_alpha_05']:.3f}"
            )

    print()
    print(
        "MULTIMODAL FOLD / PERFORMANCE CORRELATIONS"
    )

    print(
        "------------------------------------------"
    )

    ordered = correlations.copy()

    ordered[
        "absolute_spearman"
    ] = (
        ordered[
            "spearman_with_oos_ll_delta"
        ]
        .abs()
    )

    ordered = ordered.sort_values(
        "absolute_spearman",
        ascending=False,
    )

    for _, row in (
        ordered.head(
            8
        ).iterrows()
    ):
        print(
            f"{row['diagnostic']:30s} "
            f"Pearson="
            f"{row['pearson_with_oos_ll_delta']:+.3f} "
            f"Spearman="
            f"{row['spearman_with_oos_ll_delta']:+.3f}"
        )

    print()
    print(
        "Worst multimodal OOS folds"
    )

    print(
        "--------------------------"
    )

    worst = (
        fold_diagnostics
        .sort_values(
            "multimodal_minus_price_nn",
            ascending=False,
        )
        .head(
            5
        )
    )

    for _, row in (
        worst.iterrows()
    ):
        print(
            f"Fold {int(row['fold']):02d} "
            f"OOS delta="
            f"{row['multimodal_minus_price_nn']:+.6f} "
            f"mean|g|="
            f"{row['mean_total_norm']:.6f} "
            f"p99|g|="
            f"{row['p99_total_norm']:.6f} "
            f"kurt="
            f"{row['total_excess_kurtosis']:.3f} "
            f"flow/price="
            f"{row['mean_flow_to_price_ratio']:.3f}"
        )

    print()
    print(
        "Results      ->",
        RESULTS_FILE,
    )

    print(
        "Fold stats   ->",
        FOLD_OUTPUT_FILE,
    )

    print(
        "Correlations ->",
        CORRELATION_FILE,
    )


if __name__ == "__main__":
    main()
