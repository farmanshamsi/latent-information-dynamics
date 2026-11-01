from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from torch import nn

from latent_information_dynamics.models.latent.gradient_interference import (
    shared_parameters,
)


@dataclass(frozen=True)
class SharedConflictControlSnapshot:
    cosine: float
    scale: float
    applied: bool

    def as_dict(
        self,
    ) -> dict[str, object]:
        return asdict(self)


def cosine_conflict_scale(
    cosine: float,
) -> float:
    """
    Parameter-free conflict damping.

    Aligned counterfactual modality gradients leave
    the shared gradient untouched.

    For negative cosine:
        scale = 1 + cosine

    Therefore:
        cosine =  0.0 -> 1.00
        cosine = -0.25 -> 0.75
        cosine = -1.0 -> 0.00
    """

    if not math.isfinite(
        cosine
    ):
        return 1.0

    bounded = max(
        -1.0,
        min(
            1.0,
            cosine,
        ),
    )

    if bounded >= 0.0:
        return 1.0

    return 1.0 + bounded


def scale_existing_gradients(
    parameters,
    *,
    scale: float,
) -> None:
    if (
        not math.isfinite(
            scale
        )
        or scale < 0.0
        or scale > 1.0
    ):
        raise ValueError(
            "scale must be finite and in [0, 1]."
        )

    for parameter in parameters:
        if parameter.grad is None:
            continue

        parameter.grad.mul_(
            scale
        )


def apply_shared_conflict_damping(
    model: nn.Module,
    *,
    cosine: float,
) -> SharedConflictControlSnapshot:
    scale = cosine_conflict_scale(
        cosine
    )

    applied = (
        scale < 1.0
    )

    if applied:
        scale_existing_gradients(
            shared_parameters(
                model
            ),
            scale=scale,
        )

    return SharedConflictControlSnapshot(
        cosine=float(
            cosine
        ),
        scale=scale,
        applied=applied,
    )
