from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MINIMUM_SCALE = 1e-12


@dataclass(
    frozen=True,
    slots=True,
)
class FeatureStandardizer:
    """
    Feature standardization fitted exclusively
    on a legally available training sample.

    Population standard deviation (ddof=0) is used
    to match the project's existing logistic baseline
    convention.
    """

    means: tuple[float, ...]

    scales: tuple[float, ...]

    @property
    def n_features(
        self,
    ) -> int:
        return len(
            self.means
        )

    @classmethod
    def fit(
        cls,
        values: Sequence[
            Sequence[float]
        ],
    ) -> "FeatureStandardizer":
        matrix = np.asarray(
            values,
            dtype=np.float64,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Training values must form "
                "a two-dimensional matrix."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Training matrix cannot be empty."
            )

        if matrix.shape[1] == 0:
            raise ValueError(
                "Training matrix must contain "
                "at least one feature."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "Training matrix contains "
                "non-finite values."
            )

        means = np.mean(
            matrix,
            axis=0,
        )

        scales = np.std(
            matrix,
            axis=0,
            ddof=0,
        )

        scales = np.where(
            scales > MINIMUM_SCALE,
            scales,
            1.0,
        )

        return cls(
            means=tuple(
                float(value)
                for value
                in means
            ),
            scales=tuple(
                float(value)
                for value
                in scales
            ),
        )

    def transform(
        self,
        values: Sequence[
            Sequence[float]
        ],
    ) -> np.ndarray:
        matrix = np.asarray(
            values,
            dtype=np.float64,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Values must form a "
                "two-dimensional matrix."
            )

        if (
            matrix.shape[1]
            != self.n_features
        ):
            raise ValueError(
                "Feature count differs from "
                "fitted standardizer."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "Values contain non-finite "
                "entries."
            )

        means = np.asarray(
            self.means,
            dtype=np.float64,
        )

        scales = np.asarray(
            self.scales,
            dtype=np.float64,
        )

        standardized = (
            matrix - means
        ) / scales

        return standardized.astype(
            np.float32,
            copy=False,
        )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "means":
                list(
                    self.means
                ),
            "scales":
                list(
                    self.scales
                ),
            "n_features":
                self.n_features,
        }
