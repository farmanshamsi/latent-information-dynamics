import math

import torch

from latent_information_dynamics.models.latent.gradient_interference import (
    gradient_cosine,
    loss_gradient_vector,
)


def test_gradient_cosine_identical_vectors():
    left = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    assert math.isclose(
        gradient_cosine(
            left,
            left,
        ),
        1.0,
        abs_tol=1e-12,
    )


def test_gradient_cosine_opposite_vectors():
    left = torch.tensor(
        [1.0, -2.0],
        dtype=torch.float64,
    )

    right = -left

    assert math.isclose(
        gradient_cosine(
            left,
            right,
        ),
        -1.0,
        abs_tol=1e-12,
    )


def test_gradient_cosine_orthogonal_vectors():
    left = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float64,
    )

    right = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float64,
    )

    assert math.isclose(
        gradient_cosine(
            left,
            right,
        ),
        0.0,
        abs_tol=1e-12,
    )


def test_loss_gradient_vector_does_not_populate_grad():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [2.0],
            dtype=torch.float64,
        )
    )

    loss = (
        parameter ** 2
    ).sum()

    gradient = loss_gradient_vector(
        loss,
        [parameter],
    )

    assert math.isclose(
        gradient.item(),
        4.0,
        abs_tol=1e-12,
    )

    assert parameter.grad is None
