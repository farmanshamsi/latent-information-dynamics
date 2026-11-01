import math

import torch

from latent_information_dynamics.models.latent.conflict_control import (
    cosine_conflict_scale,
    scale_existing_gradients,
)


def test_positive_cosine_is_not_damped():
    assert math.isclose(
        cosine_conflict_scale(
            0.75
        ),
        1.0,
        abs_tol=1e-12,
    )


def test_negative_cosine_is_proportionally_damped():
    assert math.isclose(
        cosine_conflict_scale(
            -0.25
        ),
        0.75,
        abs_tol=1e-12,
    )


def test_maximum_conflict_zeroes_gradient_scale():
    assert math.isclose(
        cosine_conflict_scale(
            -1.0
        ),
        0.0,
        abs_tol=1e-12,
    )


def test_existing_gradient_is_scaled():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [1.0, 2.0]
        )
    )

    parameter.grad = torch.tensor(
        [4.0, -8.0]
    )

    scale_existing_gradients(
        [parameter],
        scale=0.25,
    )

    assert torch.allclose(
        parameter.grad,
        torch.tensor(
            [1.0, -2.0]
        ),
    )
