from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence


class ProbabilityMetricError(ValueError):
    """Raised when probability metrics receive invalid data."""


def _validate(
    y: Sequence[int],
    probabilities: Sequence[float],
) -> None:
    if not y:
        raise ProbabilityMetricError(
            "Evaluation sample cannot be empty."
        )

    if len(y) != len(
        probabilities
    ):
        raise ProbabilityMetricError(
            "Targets and probabilities differ in length."
        )

    if any(
        target not in (0, 1)
        for target in y
    ):
        raise ProbabilityMetricError(
            "Targets must be binary."
        )

    if any(
        (
            not math.isfinite(
                probability
            )
            or probability < 0.0
            or probability > 1.0
        )
        for probability in probabilities
    ):
        raise ProbabilityMetricError(
            "Probabilities must be finite and lie in [0, 1]."
        )


def log_loss(
    y: Sequence[int],
    probabilities: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float:
    _validate(
        y,
        probabilities,
    )

    losses = []

    for target, probability in zip(
        y,
        probabilities,
        strict=True,
    ):
        p = min(
            1.0 - epsilon,
            max(
                epsilon,
                probability,
            ),
        )

        losses.append(
            -(
                target
                * math.log(p)
                + (
                    1 - target
                )
                * math.log(
                    1.0 - p
                )
            )
        )

    return (
        sum(losses)
        / len(losses)
    )


def brier_score(
    y: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    _validate(
        y,
        probabilities,
    )

    return (
        sum(
            (
                probability
                - target
            ) ** 2
            for target, probability
            in zip(
                y,
                probabilities,
                strict=True,
            )
        )
        / len(y)
    )


def classification_accuracy(
    y: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> float:
    _validate(
        y,
        probabilities,
    )

    correct = sum(
        int(
            (
                probability
                >= threshold
            )
            == bool(target)
        )
        for target, probability
        in zip(
            y,
            probabilities,
            strict=True,
        )
    )

    return (
        correct
        / len(y)
    )


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    event_rate: float


def calibration_bins(
    y: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    _validate(
        y,
        probabilities,
    )

    if n_bins <= 0:
        raise ProbabilityMetricError(
            "n_bins must be positive."
        )

    bins: list[
        list[
            tuple[int, float]
        ]
    ] = [
        []
        for _ in range(
            n_bins
        )
    ]

    for target, probability in zip(
        y,
        probabilities,
        strict=True,
    ):
        index = min(
            n_bins - 1,
            int(
                probability
                * n_bins
            ),
        )

        bins[index].append(
            (
                target,
                probability,
            )
        )

    output = []

    for index, values in enumerate(
        bins
    ):
        if not values:
            continue

        lower = (
            index
            / n_bins
        )

        upper = (
            index + 1
        ) / n_bins

        output.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(
                    values
                ),
                mean_probability=(
                    sum(
                        probability
                        for _, probability
                        in values
                    )
                    / len(values)
                ),
                event_rate=(
                    sum(
                        target
                        for target, _
                        in values
                    )
                    / len(values)
                ),
            )
        )

    return tuple(
        output
    )
