import math

import pytest

from latent_information_dynamics.evaluation.probability import (
    brier_score,
    calibration_bins,
    log_loss,
)


def test_perfect_probabilities_have_near_zero_log_loss() -> None:
    value = log_loss(
        (
            0,
            1,
        ),
        (
            0.0,
            1.0,
        ),
    )

    assert value < 1e-9


def test_half_probability_log_loss_is_log_two() -> None:
    value = log_loss(
        (
            0,
            1,
        ),
        (
            0.5,
            0.5,
        ),
    )

    assert (
        value
        == pytest.approx(
            math.log(2.0)
        )
    )


def test_half_probability_brier_is_quarter() -> None:
    value = brier_score(
        (
            0,
            1,
        ),
        (
            0.5,
            0.5,
        ),
    )

    assert value == pytest.approx(
        0.25
    )


def test_calibration_bins_report_event_rate() -> None:
    bins = calibration_bins(
        (
            0,
            1,
            1,
            1,
        ),
        (
            0.2,
            0.2,
            0.8,
            0.8,
        ),
        n_bins=2,
    )

    assert len(bins) == 2

    assert (
        bins[0].event_rate
        == pytest.approx(
            0.5
        )
    )

    assert (
        bins[1].event_rate
        == pytest.approx(
            1.0
        )
    )
