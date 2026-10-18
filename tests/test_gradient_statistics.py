import math

from latent_information_dynamics.models.latent.gradient_statistics import (
    correlation,
    distribution_summary,
    excess_kurtosis,
    hill_tail_index,
)


def test_distribution_summary():
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
        20.0,
    ]

    result = distribution_summary(
        values
    )

    assert (
        result["count"]
        == 5
    )

    assert (
        result["maximum"]
        == 20.0
    )

    assert (
        result["p99"]
        > result["median"]
    )


def test_constant_kurtosis_is_zero():
    assert (
        excess_kurtosis(
            [
                2.0,
                2.0,
                2.0,
                2.0,
            ]
        )
        == 0.0
    )


def test_hill_tail_index_is_positive():
    values = [
        float(index)
        for index in range(
            1,
            1001,
        )
    ]

    result = hill_tail_index(
        values,
        tail_fraction=0.10,
    )

    assert math.isfinite(
        result
    )

    assert result > 0.0


def test_spearman_perfect_relationship():
    first = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    second = [
        10.0,
        20.0,
        30.0,
        40.0,
    ]

    assert math.isclose(
        correlation(
            first,
            second,
            method="spearman",
        ),
        1.0,
    )


def test_spearman_inverse_relationship():
    first = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    second = [
        40.0,
        30.0,
        20.0,
        10.0,
    ]

    assert math.isclose(
        correlation(
            first,
            second,
            method="spearman",
        ),
        -1.0,
    )
