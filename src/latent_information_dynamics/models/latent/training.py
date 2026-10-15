from __future__ import annotations

import random

from dataclasses import dataclass

import numpy as np
import torch

from torch import Tensor, nn
from torch.optim import Optimizer

from latent_information_dynamics.models.latent.diagnostics import (
    GradientSnapshot,
    capture_gradient_snapshot,
)


@dataclass(
    frozen=True,
    slots=True,
)
class TrainingStepResult:
    loss: float

    gradients: GradientSnapshot


def set_research_seed(
    seed: int,
) -> None:
    """
    Set the random states needed by the
    current research implementation.

    Outer experiments will initially use CPU
    execution for reproducibility.
    """

    if seed < 0:
        raise ValueError(
            "seed cannot be negative."
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def binary_training_step(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    price_x: Tensor,
    flow_x: Tensor,
    target: Tensor,
) -> TrainingStepResult:
    """
    Execute one optimization step while recording
    pre-update gradient diagnostics.
    """

    if target.ndim != 1:
        raise ValueError(
            "target must have shape (batch,)."
        )

    if (
        price_x.shape[0]
        != target.shape[0]
    ):
        raise ValueError(
            "Price batch and target batch differ."
        )

    if (
        flow_x.shape[0]
        != target.shape[0]
    ):
        raise ValueError(
            "Flow batch and target batch differ."
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        price_x,
        flow_x,
    )

    if (
        logits.shape
        != target.shape
    ):
        raise ValueError(
            "Model logit shape does not match "
            "target shape."
        )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    loss = criterion(
        logits,
        target,
    )

    loss.backward()

    gradients = (
        capture_gradient_snapshot(
            model
        )
    )

    optimizer.step()

    return TrainingStepResult(
        loss=
            float(
                loss
                .detach()
                .item()
            ),
        gradients=
            gradients,
    )
