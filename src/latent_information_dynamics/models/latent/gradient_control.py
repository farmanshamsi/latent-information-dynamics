from __future__ import annotations

import math

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from latent_information_dynamics.models.latent.diagnostics import (
    parameter_gradient_norm,
)


@dataclass(
    frozen=True,
    slots=True,
)
class GradientControlSnapshot:
    """
    Effect of a gradient transformation applied after
    backward() and before optimizer.step().
    """

    raw_total_norm: float

    effective_total_norm: float

    scale: float

    clipped: bool

    def as_dict(
        self,
    ) -> dict[str, float | bool]:
        return {
            "raw_total_norm":
                self.raw_total_norm,
            "effective_total_norm":
                self.effective_total_norm,
            "gradient_scale":
                self.scale,
            "gradient_clipped":
                self.clipped,
        }


def clip_gradients_by_global_norm(
    parameters: Iterable[
        nn.Parameter
    ],
    *,
    max_norm: float,
    epsilon: float = 1e-12,
) -> GradientControlSnapshot:
    """
    Clip the complete parameter-gradient vector by its
    global Euclidean norm.

    The same multiplicative scale is applied to every
    available parameter gradient, preserving the raw
    gradient direction.

    This function deliberately records both raw and
    effective norms so optimization diagnostics remain
    interpretable.
    """

    if (
        not math.isfinite(
            max_norm
        )
        or max_norm <= 0.0
    ):
        raise ValueError(
            "max_norm must be positive and finite."
        )

    if (
        not math.isfinite(
            epsilon
        )
        or epsilon <= 0.0
    ):
        raise ValueError(
            "epsilon must be positive and finite."
        )

    parameter_tuple = tuple(
        parameters
    )

    raw_norm = (
        parameter_gradient_norm(
            parameter_tuple
        )
    )

    if not math.isfinite(
        raw_norm
    ):
        raise RuntimeError(
            "Cannot clip non-finite gradient norm."
        )

    if (
        raw_norm == 0.0
        or raw_norm <= max_norm
    ):
        return GradientControlSnapshot(
            raw_total_norm=
                raw_norm,
            effective_total_norm=
                raw_norm,
            scale=
                1.0,
            clipped=
                False,
        )

    scale = (
        max_norm
        / max(
            raw_norm,
            epsilon,
        )
    )

    for parameter in parameter_tuple:
        gradient = parameter.grad

        if gradient is None:
            continue

        gradient.mul_(
            scale
        )

    effective_norm = (
        parameter_gradient_norm(
            parameter_tuple
        )
    )

    return GradientControlSnapshot(
        raw_total_norm=
            raw_norm,
        effective_total_norm=
            effective_norm,
        scale=
            scale,
        clipped=
            True,
    )
