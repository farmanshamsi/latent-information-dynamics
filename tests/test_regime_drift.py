import math

from latent_information_dynamics.models.latent.regime_drift import (
    covariance_shift,
    log_scale_shift,
    standardized_mean_shift,
)


def test_identical_regimes_have_zero_mean_shift():
    sample = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ]

    result = standardized_mean_shift(
        sample,
        sample,
    )

    assert math.isclose(
        result["mean_absolute"],
        0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        result["rms"],
        0.0,
        abs_tol=1e-12,
    )


def test_mean_shift_detects_location_change():
    reference = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]

    comparison = [
        [10.0],
        [11.0],
        [12.0],
        [13.0],
    ]

    result = standardized_mean_shift(
        reference,
        comparison,
    )

    assert result[
        "mean_absolute"
    ] > 1.0


def test_log_scale_shift_detects_scale_change():
    reference = [
        [-1.0],
        [0.0],
        [1.0],
        [0.0],
    ]

    comparison = [
        [-10.0],
        [0.0],
        [10.0],
        [0.0],
    ]

    result = log_scale_shift(
        reference,
        comparison,
    )

    assert result[
        "mean_absolute"
    ] > 1.0


def test_identical_covariance_has_zero_shift():
    sample = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
    ]

    assert math.isclose(
        covariance_shift(
            sample,
            sample,
        ),
        0.0,
        abs_tol=1e-12,
    )
