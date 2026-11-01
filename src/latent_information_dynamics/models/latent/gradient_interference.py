from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable

import torch
from torch import nn


EPSILON = 1e-12


@dataclass(frozen=True)
class SharedGradientInterference:
    full_shared_norm: float
    price_context_shared_norm: float
    flow_context_shared_norm: float

    price_flow_cosine: float
    full_price_cosine: float
    full_flow_cosine: float

    price_context_loss: float
    flow_context_loss: float

    price_context_zero_gradient: bool
    flow_context_zero_gradient: bool

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)


def _parameter_tuple(
    parameters: Iterable[
        nn.Parameter
    ],
) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for parameter
        in parameters
        if parameter.requires_grad
    )


def shared_parameters(
    model: nn.Module,
) -> tuple[nn.Parameter, ...]:
    if not hasattr(
        model,
        "fusion",
    ):
        raise AttributeError(
            "Model must expose fusion."
        )

    if not hasattr(
        model,
        "prediction_head",
    ):
        raise AttributeError(
            "Model must expose prediction_head."
        )

    return _parameter_tuple(
        tuple(
            model.fusion.parameters()
        )
        + tuple(
            model.prediction_head.parameters()
        )
    )


def existing_gradient_vector(
    parameters: Iterable[
        nn.Parameter
    ],
) -> torch.Tensor:
    parameters = _parameter_tuple(
        parameters
    )

    if not parameters:
        raise ValueError(
            "No trainable parameters."
        )

    pieces = []

    for parameter in parameters:
        if parameter.grad is None:
            piece = torch.zeros_like(
                parameter
            )
        else:
            piece = parameter.grad

        pieces.append(
            piece
            .detach()
            .reshape(-1)
            .to(
                dtype=torch.float64,
                device="cpu",
            )
        )

    return torch.cat(
        pieces
    )


def loss_gradient_vector(
    loss: torch.Tensor,
    parameters: Iterable[
        nn.Parameter
    ],
) -> torch.Tensor:
    parameters = _parameter_tuple(
        parameters
    )

    if not parameters:
        raise ValueError(
            "No trainable parameters."
        )

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    pieces = []

    for parameter, gradient in zip(
        parameters,
        gradients,
        strict=True,
    ):
        if gradient is None:
            piece = torch.zeros_like(
                parameter
            )
        else:
            piece = gradient

        pieces.append(
            piece
            .detach()
            .reshape(-1)
            .to(
                dtype=torch.float64,
                device="cpu",
            )
        )

    return torch.cat(
        pieces
    )


def vector_norm(
    vector: torch.Tensor,
) -> float:
    return float(
        torch.linalg.vector_norm(
            vector
        ).item()
    )


def gradient_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float = EPSILON,
) -> float:
    if left.shape != right.shape:
        raise ValueError(
            "Gradient vectors must have "
            "matching dimensions."
        )

    left_norm = vector_norm(
        left
    )

    right_norm = vector_norm(
        right
    )

    if (
        left_norm <= epsilon
        or right_norm <= epsilon
    ):
        return float(
            "nan"
        )

    cosine = (
        torch.dot(
            left,
            right,
        ).item()
        / (
            left_norm
            * right_norm
        )
    )

    return float(
        max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )
    )


def capture_shared_gradient_interference(
    model: nn.Module,
    price_x: torch.Tensor,
    flow_x: torch.Tensor,
    target: torch.Tensor,
) -> SharedGradientInterference:
    """
    Assumes the full multimodal loss has already been
    backpropagated and model parameter .grad fields hold
    the actual training gradient.

    Counterfactual Price-context and Flow-context losses
    are differentiated only with respect to shared
    fusion/head parameters. autograd.grad does not alter
    the existing training gradients.
    """

    parameters = shared_parameters(
        model
    )

    full_gradient = (
        existing_gradient_vector(
            parameters
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    zero_flow = torch.zeros_like(
        flow_x
    )

    zero_price = torch.zeros_like(
        price_x
    )

    price_logits = model(
        price_x,
        zero_flow,
    )

    price_loss = criterion(
        price_logits,
        target,
    )

    price_gradient = (
        loss_gradient_vector(
            price_loss,
            parameters,
        )
    )

    flow_logits = model(
        zero_price,
        flow_x,
    )

    flow_loss = criterion(
        flow_logits,
        target,
    )

    flow_gradient = (
        loss_gradient_vector(
            flow_loss,
            parameters,
        )
    )

    price_norm = vector_norm(
        price_gradient
    )

    flow_norm = vector_norm(
        flow_gradient
    )

    return SharedGradientInterference(
        full_shared_norm=
            vector_norm(
                full_gradient
            ),
        price_context_shared_norm=
            price_norm,
        flow_context_shared_norm=
            flow_norm,
        price_flow_cosine=
            gradient_cosine(
                price_gradient,
                flow_gradient,
            ),
        full_price_cosine=
            gradient_cosine(
                full_gradient,
                price_gradient,
            ),
        full_flow_cosine=
            gradient_cosine(
                full_gradient,
                flow_gradient,
            ),
        price_context_loss=
            float(
                price_loss
                .detach()
                .item()
            ),
        flow_context_loss=
            float(
                flow_loss
                .detach()
                .item()
            ),
        price_context_zero_gradient=
            price_norm
            <= EPSILON,
        flow_context_zero_gradient=
            flow_norm
            <= EPSILON,
    )
