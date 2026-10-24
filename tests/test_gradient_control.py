import math

import pytest
import torch

from latent_information_dynamics.models.latent.gradient_control import (
    clip_gradients_by_global_norm,
)


def test_global_norm_clipping_reduces_gradient():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0, 0.0]
        )
    )

    parameter.grad = torch.tensor(
        [3.0, 4.0]
    )

    result = (
        clip_gradients_by_global_norm(
            [parameter],
            max_norm=2.0,
        )
    )

    assert math.isclose(
        result.raw_total_norm,
        5.0,
        rel_tol=1e-7,
    )

    assert math.isclose(
        result.effective_total_norm,
        2.0,
        rel_tol=1e-6,
    )

    assert math.isclose(
        result.scale,
        0.4,
        rel_tol=1e-7,
    )

    assert result.clipped


def test_gradient_below_threshold_is_unchanged():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0, 0.0]
        )
    )

    parameter.grad = torch.tensor(
        [0.3, 0.4]
    )

    before = (
        parameter.grad
        .detach()
        .clone()
    )

    result = (
        clip_gradients_by_global_norm(
            [parameter],
            max_norm=1.0,
        )
    )

    assert not result.clipped

    assert result.scale == 1.0

    assert torch.equal(
        parameter.grad,
        before,
    )


def test_zero_gradient_is_safe():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0]
        )
    )

    parameter.grad = torch.tensor(
        [0.0]
    )

    result = (
        clip_gradients_by_global_norm(
            [parameter],
            max_norm=0.1,
        )
    )

    assert (
        result.raw_total_norm
        == 0.0
    )

    assert (
        result.effective_total_norm
        == 0.0
    )

    assert not result.clipped


def test_invalid_clip_threshold_rejected():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [0.0]
        )
    )

    parameter.grad = torch.tensor(
        [1.0]
    )

    with pytest.raises(
        ValueError
    ):
        clip_gradients_by_global_norm(
            [parameter],
            max_norm=0.0,
        )
