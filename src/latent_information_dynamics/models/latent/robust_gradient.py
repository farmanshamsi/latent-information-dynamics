from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class RobustGradientSnapshot:
    groups: int
    robust_gradient_norm: float
    mean_group_gradient_norm: float
    median_group_gradient_norm: float
    maximum_group_gradient_norm: float
    max_to_median_norm_ratio: float

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)


def coordinatewise_median(
    tensors: list[torch.Tensor],
) -> torch.Tensor:
    if len(tensors) < 3:
        raise ValueError(
            "At least three tensors are required."
        )

    if len(tensors) % 2 == 0:
        raise ValueError(
            "An odd number of tensors is required."
        )

    shape = tensors[0].shape

    if any(
        tensor.shape != shape
        for tensor in tensors
    ):
        raise ValueError(
            "All tensors must have matching shapes."
        )

    stacked = torch.stack(
        tensors,
        dim=0,
    )

    return torch.median(
        stacked,
        dim=0,
    ).values


def apply_coordinatewise_median_of_means(
    *,
    losses: list[torch.Tensor],
    parameters,
) -> RobustGradientSnapshot:
    """
    Each loss is the mean loss on one block of the
    current minibatch.

    We compute one gradient vector per block and use
    the coordinatewise median across block-mean
    gradients as the stochastic-gradient estimate.
    """

    if len(losses) < 3:
        raise ValueError(
            "At least three blocks are required."
        )

    if len(losses) % 2 == 0:
        raise ValueError(
            "An odd number of blocks is required."
        )

    parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.requires_grad
    )

    if not parameters:
        raise ValueError(
            "No trainable parameters."
        )

    block_gradients = []

    for loss in losses:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        completed = []

        for parameter, gradient in zip(
            parameters,
            gradients,
            strict=True,
        ):
            if gradient is None:
                gradient = torch.zeros_like(
                    parameter
                )

            completed.append(
                gradient.detach()
            )

        block_gradients.append(
            tuple(
                completed
            )
        )

    group_squared_norms = [
        0.0
        for _ in block_gradients
    ]

    robust_squared_norm = 0.0

    for parameter_index, parameter in enumerate(
        parameters
    ):
        gradients = [
            block[
                parameter_index
            ]
            for block in block_gradients
        ]

        robust_gradient = (
            coordinatewise_median(
                gradients
            )
        )

        parameter.grad = (
            robust_gradient
            .detach()
            .clone()
        )

        robust_squared_norm += float(
            torch.sum(
                robust_gradient
                .to(
                    dtype=torch.float64
                )
                ** 2
            ).item()
        )

        for group_index, gradient in enumerate(
            gradients
        ):
            group_squared_norms[
                group_index
            ] += float(
                torch.sum(
                    gradient
                    .to(
                        dtype=torch.float64
                    )
                    ** 2
                ).item()
            )

    group_norms = [
        math.sqrt(
            value
        )
        for value in group_squared_norms
    ]

    robust_norm = math.sqrt(
        robust_squared_norm
    )

    sorted_norms = sorted(
        group_norms
    )

    median_norm = sorted_norms[
        len(
            sorted_norms
        )
        // 2
    ]

    maximum_norm = max(
        group_norms
    )

    ratio = (
        maximum_norm
        / median_norm
        if median_norm > 0.0
        else float(
            "inf"
        )
    )

    return RobustGradientSnapshot(
        groups=len(
            losses
        ),
        robust_gradient_norm=
            robust_norm,
        mean_group_gradient_norm=
            float(
                sum(
                    group_norms
                )
                / len(
                    group_norms
                )
            ),
        median_group_gradient_norm=
            median_norm,
        maximum_group_gradient_norm=
            maximum_norm,
        max_to_median_norm_ratio=
            ratio,
    )
