from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Iterable, Sequence


class BinaryModelError(ValueError):
    """Raised when a binary probability model cannot be fitted."""


def _clip_probability(
    value: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    return min(
        1.0 - epsilon,
        max(
            epsilon,
            value,
        ),
    )


def _sigmoid(
    value: float,
) -> float:
    if value >= 0:
        z = math.exp(
            -value
        )

        return 1.0 / (
            1.0 + z
        )

    z = math.exp(
        value
    )

    return z / (
        1.0 + z
    )


def _solve_linear_system(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    """
    Solve Ax=b using Gaussian elimination with partial pivoting.

    Kept dependency-free because this is a small research baseline,
    not a high-dimensional production optimizer.
    """

    n = len(
        vector
    )

    augmented = [
        [
            *matrix[row],
            vector[row],
        ]
        for row in range(n)
    ]

    for column in range(n):
        pivot = max(
            range(
                column,
                n,
            ),
            key=lambda row:
                abs(
                    augmented[row][column]
                ),
        )

        if abs(
            augmented[pivot][column]
        ) < 1e-14:
            raise BinaryModelError(
                "Logistic Hessian is singular."
            )

        augmented[
            column
        ], augmented[
            pivot
        ] = (
            augmented[pivot],
            augmented[column],
        )

        pivot_value = (
            augmented[column][column]
        )

        for item in range(
            column,
            n + 1,
        ):
            augmented[column][item] /= (
                pivot_value
            )

        for row in range(n):
            if row == column:
                continue

            factor = (
                augmented[row][column]
            )

            if factor == 0:
                continue

            for item in range(
                column,
                n + 1,
            ):
                augmented[row][item] -= (
                    factor
                    * augmented[column][item]
                )

    return [
        augmented[row][n]
        for row in range(n)
    ]


@dataclass(slots=True)
class ConstantProbabilityModel:
    """
    Unconditional Bernoulli benchmark.

    Estimates P(Y=1) from the legally available training sample.
    """

    probability_: float | None = None

    def fit(
        self,
        X: Sequence[
            Sequence[float]
        ],
        y: Sequence[int],
    ) -> "ConstantProbabilityModel":
        if not y:
            raise BinaryModelError(
                "Training labels cannot be empty."
            )

        if any(
            value not in (0, 1)
            for value in y
        ):
            raise BinaryModelError(
                "Labels must be binary."
            )

        self.probability_ = (
            sum(y)
            / len(y)
        )

        return self

    def predict_proba(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> tuple[float, ...]:
        if self.probability_ is None:
            raise BinaryModelError(
                "Model must be fitted before prediction."
            )

        probability = _clip_probability(
            self.probability_
        )

        return tuple(
            probability
            for _ in X
        )


@dataclass(slots=True)
class LogisticRegressionBinary:
    """
    Small dependency-free L2-regularized logistic regression.

    Features are standardized using training data only.

    Newton/IRLS optimization is sufficient for the low-dimensional
    benchmark models used in this project.
    """

    l2: float = 1.0
    max_iterations: int = 100
    tolerance: float = 1e-8

    means_: tuple[float, ...] | None = None
    scales_: tuple[float, ...] | None = None
    coefficients_: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.l2 < 0:
            raise ValueError(
                "l2 cannot be negative."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        if self.tolerance <= 0:
            raise ValueError(
                "tolerance must be positive."
            )

    def _validate_training_data(
        self,
        X: Sequence[
            Sequence[float]
        ],
        y: Sequence[int],
    ) -> int:
        if not X:
            raise BinaryModelError(
                "Training matrix cannot be empty."
            )

        if len(X) != len(y):
            raise BinaryModelError(
                "X and y lengths differ."
            )

        width = len(
            X[0]
        )

        if width == 0:
            raise BinaryModelError(
                "At least one feature is required."
            )

        for row in X:
            if len(row) != width:
                raise BinaryModelError(
                    "Training matrix rows have inconsistent widths."
                )

            if any(
                not math.isfinite(
                    float(value)
                )
                for value in row
            ):
                raise BinaryModelError(
                    "Training features must be finite."
                )

        if any(
            value not in (0, 1)
            for value in y
        ):
            raise BinaryModelError(
                "Labels must be binary."
            )

        if len(
            set(y)
        ) < 2:
            raise BinaryModelError(
                "Logistic regression requires both classes."
            )

        return width

    def _standardize_fit(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> list[list[float]]:
        n = len(
            X
        )

        width = len(
            X[0]
        )

        means = []

        scales = []

        for column in range(width):
            values = [
                float(
                    row[column]
                )
                for row in X
            ]

            mean = (
                sum(values)
                / n
            )

            variance = (
                sum(
                    (
                        value - mean
                    ) ** 2
                    for value in values
                )
                / n
            )

            scale = math.sqrt(
                variance
            )

            if scale < 1e-12:
                scale = 1.0

            means.append(
                mean
            )

            scales.append(
                scale
            )

        self.means_ = tuple(
            means
        )

        self.scales_ = tuple(
            scales
        )

        return [
            [
                (
                    float(value)
                    - means[column]
                )
                / scales[column]
                for column, value
                in enumerate(row)
            ]
            for row in X
        ]

    def _standardize_predict(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> list[list[float]]:
        if (
            self.means_ is None
            or self.scales_ is None
        ):
            raise BinaryModelError(
                "Model has not been fitted."
            )

        width = len(
            self.means_
        )

        output = []

        for row in X:
            if len(row) != width:
                raise BinaryModelError(
                    "Prediction feature width differs from training."
                )

            output.append(
                [
                    (
                        float(value)
                        - self.means_[column]
                    )
                    / self.scales_[column]
                    for column, value
                    in enumerate(row)
                ]
            )

        return output

    def fit(
        self,
        X: Sequence[
            Sequence[float]
        ],
        y: Sequence[int],
    ) -> "LogisticRegressionBinary":
        width = self._validate_training_data(
            X,
            y,
        )

        standardized = self._standardize_fit(
            X
        )

        design = [
            [
                1.0,
                *row,
            ]
            for row in standardized
        ]

        dimension = (
            width + 1
        )

        beta = [
            0.0
            for _ in range(
                dimension
            )
        ]

        for _ in range(
            self.max_iterations
        ):
            probabilities = [
                _sigmoid(
                    sum(
                        coefficient * value
                        for coefficient, value
                        in zip(
                            beta,
                            row,
                            strict=True,
                        )
                    )
                )
                for row in design
            ]

            gradient = [
                0.0
                for _ in range(
                    dimension
                )
            ]

            hessian = [
                [
                    0.0
                    for _ in range(
                        dimension
                    )
                ]
                for _ in range(
                    dimension
                )
            ]

            for row, target, probability in zip(
                design,
                y,
                probabilities,
                strict=True,
            ):
                residual = (
                    target
                    - probability
                )

                weight = (
                    probability
                    * (
                        1.0
                        - probability
                    )
                )

                for j in range(
                    dimension
                ):
                    gradient[j] += (
                        row[j]
                        * residual
                    )

                    for k in range(
                        dimension
                    ):
                        hessian[j][k] += (
                            weight
                            * row[j]
                            * row[k]
                        )

            for j in range(
                1,
                dimension,
            ):
                gradient[j] -= (
                    self.l2
                    * beta[j]
                )

                hessian[j][j] += (
                    self.l2
                )

            # Tiny intercept stabilization.
            hessian[0][0] += 1e-12

            step = _solve_linear_system(
                hessian,
                gradient,
            )

            beta = [
                coefficient
                + update
                for coefficient, update
                in zip(
                    beta,
                    step,
                    strict=True,
                )
            ]

            if max(
                abs(value)
                for value in step
            ) < self.tolerance:
                break

        self.coefficients_ = tuple(
            beta
        )

        return self

    def predict_proba(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> tuple[float, ...]:
        if self.coefficients_ is None:
            raise BinaryModelError(
                "Model must be fitted before prediction."
            )

        standardized = (
            self._standardize_predict(
                X
            )
        )

        output = []

        for row in standardized:
            design_row = [
                1.0,
                *row,
            ]

            score = sum(
                coefficient * value
                for coefficient, value
                in zip(
                    self.coefficients_,
                    design_row,
                    strict=True,
                )
            )

            output.append(
                _clip_probability(
                    _sigmoid(
                        score
                    )
                )
            )

        return tuple(
            output
        )
