from __future__ import annotations

from collections.abc import Sequence

import numpy as np


EPSILON = 1e-12


def _matrix(
    values: Sequence[
        Sequence[float]
    ],
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim != 2
        or array.shape[0] < 2
        or array.shape[1] < 1
    ):
        raise ValueError(
            "Expected a non-empty "
            "two-dimensional sample."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "Regime-drift inputs must be finite."
        )

    return array


def standardized_mean_shift(
    reference: Sequence[
        Sequence[float]
    ],
    comparison: Sequence[
        Sequence[float]
    ],
) -> dict[str, float]:
    """
    Shift in feature means measured in units of
    reference-regime standard deviation.
    """

    reference_array = _matrix(
        reference
    )

    comparison_array = _matrix(
        comparison
    )

    if (
        reference_array.shape[1]
        != comparison_array.shape[1]
    ):
        raise ValueError(
            "Feature dimensions must match."
        )

    reference_mean = np.mean(
        reference_array,
        axis=0,
    )

    comparison_mean = np.mean(
        comparison_array,
        axis=0,
    )

    scale = np.std(
        reference_array,
        axis=0,
        ddof=0,
    )

    scale = np.where(
        scale > EPSILON,
        scale,
        1.0,
    )

    shift = (
        comparison_mean
        - reference_mean
    ) / scale

    absolute = np.abs(
        shift
    )

    return {
        "mean_absolute":
            float(
                np.mean(
                    absolute
                )
            ),
        "rms":
            float(
                np.sqrt(
                    np.mean(
                        shift ** 2
                    )
                )
            ),
        "maximum_absolute":
            float(
                np.max(
                    absolute
                )
            ),
    }


def log_scale_shift(
    reference: Sequence[
        Sequence[float]
    ],
    comparison: Sequence[
        Sequence[float]
    ],
) -> dict[str, float]:
    """
    Absolute log-ratio change in feature scale.
    """

    reference_array = _matrix(
        reference
    )

    comparison_array = _matrix(
        comparison
    )

    if (
        reference_array.shape[1]
        != comparison_array.shape[1]
    ):
        raise ValueError(
            "Feature dimensions must match."
        )

    reference_scale = np.std(
        reference_array,
        axis=0,
        ddof=0,
    )

    comparison_scale = np.std(
        comparison_array,
        axis=0,
        ddof=0,
    )

    reference_scale = np.maximum(
        reference_scale,
        EPSILON,
    )

    comparison_scale = np.maximum(
        comparison_scale,
        EPSILON,
    )

    shift = np.abs(
        np.log(
            comparison_scale
            / reference_scale
        )
    )

    return {
        "mean_absolute":
            float(
                np.mean(
                    shift
                )
            ),
        "rms":
            float(
                np.sqrt(
                    np.mean(
                        shift ** 2
                    )
                )
            ),
        "maximum_absolute":
            float(
                np.max(
                    shift
                )
            ),
    }


def covariance_shift(
    reference: Sequence[
        Sequence[float]
    ],
    comparison: Sequence[
        Sequence[float]
    ],
) -> float:
    """
    Frobenius covariance shift after scaling both
    regimes using reference-regime feature scales.

    Returned value is normalized by feature count.
    """

    reference_array = _matrix(
        reference
    )

    comparison_array = _matrix(
        comparison
    )

    if (
        reference_array.shape[1]
        != comparison_array.shape[1]
    ):
        raise ValueError(
            "Feature dimensions must match."
        )

    reference_scale = np.std(
        reference_array,
        axis=0,
        ddof=0,
    )

    reference_scale = np.where(
        reference_scale > EPSILON,
        reference_scale,
        1.0,
    )

    reference_centered = (
        reference_array
        - np.mean(
            reference_array,
            axis=0,
        )
    ) / reference_scale

    comparison_centered = (
        comparison_array
        - np.mean(
            comparison_array,
            axis=0,
        )
    ) / reference_scale

    reference_covariance = (
        reference_centered.T
        @ reference_centered
        / reference_centered.shape[0]
    )

    comparison_covariance = (
        comparison_centered.T
        @ comparison_centered
        / comparison_centered.shape[0]
    )

    difference = (
        comparison_covariance
        - reference_covariance
    )

    dimension = (
        reference_array.shape[1]
    )

    return float(
        np.linalg.norm(
            difference,
            ord="fro",
        )
        / dimension
    )
