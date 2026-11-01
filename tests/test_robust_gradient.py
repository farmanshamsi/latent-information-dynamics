import math

import pytest
import torch

from latent_information_dynamics.models.latent.robust_gradient import (
    apply_coordinatewise_median_of_means,
    coordinatewise_median,
)


def test_coordinatewise_median_rejects_even_count():
    with pytest.raises(
        ValueError
    ):
        coordinatewise_median(
            [
                torch.tensor([1.0]),
                torch.tensor([2.0]),
            ]
        )


def test_coordinatewise_median_rejects_outlier():
    result = coordinatewise_median(
        [
            torch.tensor(
                [1.0, 2.0]
            ),
            torch.tensor(
                [1.1, 1.9]
            ),
            torch.tensor(
                [100.0, -100.0]
            ),
        ]
    )

    assert torch.allclose(
        result,
        torch.tensor(
            [1.1, 1.9]
        ),
    )


def test_median_of_means_gradient_rejects_gradient_outlier():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0]
        )
    )

    losses = [
        (
            parameter - 1.0
        ).pow(
            2
        ).mean(),
        (
            parameter - 1.0
        ).pow(
            2
        ).mean(),
        (
            parameter - 100.0
        ).pow(
            2
        ).mean(),
    ]

    snapshot = (
        apply_coordinatewise_median_of_means(
            losses=losses,
            parameters=[
                parameter
            ],
        )
    )

    assert torch.allclose(
        parameter.grad,
        torch.tensor(
            [-2.0]
        ),
    )

    assert (
        snapshot
        .maximum_group_gradient_norm
        > snapshot
        .median_group_gradient_norm
    )


def test_robust_gradient_snapshot_norm():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0]
        )
    )

    losses = [
        (
            parameter - value
        ).pow(
            2
        ).mean()
        for value in (
            1.0,
            1.0,
            1.0,
            1.0,
            20.0,
        )
    ]

    snapshot = (
        apply_coordinatewise_median_of_means(
            losses=losses,
            parameters=[
                parameter
            ],
        )
    )

    assert math.isclose(
        snapshot.robust_gradient_norm,
        2.0,
        abs_tol=1e-12,
    )
