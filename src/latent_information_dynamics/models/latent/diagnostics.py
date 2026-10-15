from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(
    frozen=True,
    slots=True,
)
class GradientSnapshot:
    """
    Gradient diagnostics recorded after backward()
    and before optimizer.step().
    """

    total_norm: float

    price_encoder_norm: float

    flow_encoder_norm: float

    fusion_norm: float

    prediction_head_norm: float

    max_abs_gradient: float

    nonfinite_gradient_elements: int

    def as_dict(
        self,
    ) -> dict[str, float | int]:
        return {
            "total_norm":
                self.total_norm,
            "price_encoder_norm":
                self.price_encoder_norm,
            "flow_encoder_norm":
                self.flow_encoder_norm,
            "fusion_norm":
                self.fusion_norm,
            "prediction_head_norm":
                self.prediction_head_norm,
            "max_abs_gradient":
                self.max_abs_gradient,
            "nonfinite_gradient_elements":
                self.nonfinite_gradient_elements,
        }


def parameter_gradient_norm(
    parameters: Iterable[
        nn.Parameter
    ],
) -> float:
    """
    Euclidean norm across every available
    parameter gradient.
    """

    squared_norm = 0.0

    for parameter in parameters:
        gradient = (
            parameter.grad
        )

        if gradient is None:
            continue

        detached = (
            gradient
            .detach()
            .double()
        )

        squared_norm += float(
            torch.sum(
                detached
                * detached
            ).item()
        )

    return math.sqrt(
        squared_norm
    )


def _gradient_extremes(
    parameters: Iterable[
        nn.Parameter
    ],
) -> tuple[
    float,
    int,
]:
    max_abs = 0.0

    nonfinite = 0

    for parameter in parameters:
        gradient = (
            parameter.grad
        )

        if gradient is None:
            continue

        detached = (
            gradient.detach()
        )

        finite_mask = (
            torch.isfinite(
                detached
            )
        )

        nonfinite += int(
            (
                ~finite_mask
            )
            .sum()
            .item()
        )

        if finite_mask.any():
            finite_values = (
                detached[
                    finite_mask
                ]
                .abs()
            )

            current_max = float(
                finite_values
                .max()
                .item()
            )

            max_abs = max(
                max_abs,
                current_max,
            )

    return (
        max_abs,
        nonfinite,
    )


def capture_gradient_snapshot(
    model: nn.Module,
) -> GradientSnapshot:
    """
    Capture total and architecture-specific
    gradient norms.

    The model must expose:
      price_encoder
      flow_encoder
      fusion
      prediction_head
    """

    required = (
        "price_encoder",
        "flow_encoder",
        "fusion",
        "prediction_head",
    )

    missing = [
        name
        for name in required
        if not hasattr(
            model,
            name,
        )
    ]

    if missing:
        raise ValueError(
            "Model does not expose expected "
            "gradient groups: "
            f"{missing}"
        )

    all_parameters = tuple(
        model.parameters()
    )

    (
        max_abs_gradient,
        nonfinite_gradient_elements,
    ) = _gradient_extremes(
        all_parameters
    )

    return GradientSnapshot(
        total_norm=
            parameter_gradient_norm(
                all_parameters
            ),
        price_encoder_norm=
            parameter_gradient_norm(
                model
                .price_encoder
                .parameters()
            ),
        flow_encoder_norm=
            parameter_gradient_norm(
                model
                .flow_encoder
                .parameters()
            ),
        fusion_norm=
            parameter_gradient_norm(
                model
                .fusion
                .parameters()
            ),
        prediction_head_norm=
            parameter_gradient_norm(
                model
                .prediction_head
                .parameters()
            ),
        max_abs_gradient=
            max_abs_gradient,
        nonfinite_gradient_elements=
            nonfinite_gradient_elements,
    )
