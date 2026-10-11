from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol, Sequence

from latent_information_dynamics.datasets.supervised import (
    SupervisedDataset,
)

from .probability import (
    brier_score,
    classification_accuracy,
    log_loss,
)


class ProbabilisticBinaryModel(Protocol):

    def fit(
        self,
        X: Sequence[
            Sequence[float]
        ],
        y: Sequence[int],
    ) -> "ProbabilisticBinaryModel":
        ...

    def predict_proba(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> tuple[float, ...]:
        ...


class WalkForwardError(ValueError):
    """Raised when chronological evaluation cannot be constructed."""


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    training_cutoff: datetime
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    fold: int
    decision_time: datetime
    target: int
    probability: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    feature_names: tuple[str, ...]
    forecasts: tuple[
        ForecastRecord,
        ...
    ]

    log_loss: float
    brier_score: float
    accuracy: float

    @property
    def targets(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            forecast.target
            for forecast in self.forecasts
        )

    @property
    def probabilities(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            forecast.probability
            for forecast in self.forecasts
        )

    @property
    def decision_times(
        self,
    ) -> tuple[datetime, ...]:
        return tuple(
            forecast.decision_time
            for forecast in self.forecasts
        )


def expanding_walk_forward_folds(
    dataset: SupervisedDataset,
    *,
    min_train_observations: int,
    test_observations: int,
) -> tuple[WalkForwardFold, ...]:
    """
    Build expanding chronological folds.

    A training observation is legal only when its realized label was
    already available by the first decision time of the test fold.
    """

    if min_train_observations <= 0:
        raise WalkForwardError(
            "min_train_observations must be positive."
        )

    if test_observations <= 0:
        raise WalkForwardError(
            "test_observations must be positive."
        )

    examples = dataset.examples

    folds = []

    candidate = (
        min_train_observations
    )

    fold_number = 0

    while candidate < len(
        examples
    ):
        test_start_time = (
            examples[
                candidate
            ].decision_time
        )

        train_indices = tuple(
            index
            for index, example
            in enumerate(
                examples[
                    :candidate
                ]
            )
            if (
                example.label_available_time
                <= test_start_time
            )
        )

        if len(
            train_indices
        ) < min_train_observations:
            candidate += 1
            continue

        test_end = min(
            candidate
            + test_observations,
            len(
                examples
            ),
        )

        test_indices = tuple(
            range(
                candidate,
                test_end,
            )
        )

        folds.append(
            WalkForwardFold(
                fold=fold_number,
                training_cutoff=
                    test_start_time,
                train_indices=
                    train_indices,
                test_indices=
                    test_indices,
            )
        )

        fold_number += 1

        candidate = (
            test_end
        )

    if not folds:
        raise WalkForwardError(
            "No legal walk-forward folds could be created."
        )

    return tuple(
        folds
    )


def evaluate_walk_forward(
    dataset: SupervisedDataset,
    *,
    feature_names: Sequence[str],
    model_factory: Callable[
        [],
        ProbabilisticBinaryModel,
    ],
    min_train_observations: int,
    test_observations: int,
) -> EvaluationResult:
    """
    Fit only on historically available labels and produce strictly
    out-of-sample probabilities.
    """

    selected = tuple(
        feature_names
    )

    matrix = dataset.matrix(
        features=selected
    )

    folds = expanding_walk_forward_folds(
        dataset,
        min_train_observations=
            min_train_observations,
        test_observations=
            test_observations,
    )

    forecasts = []

    for fold in folds:
        train_X = tuple(
            matrix.X[index]
            for index
            in fold.train_indices
        )

        train_y = tuple(
            matrix.y[index]
            for index
            in fold.train_indices
        )

        test_X = tuple(
            matrix.X[index]
            for index
            in fold.test_indices
        )

        model = (
            model_factory()
        )

        model.fit(
            train_X,
            train_y,
        )

        probabilities = (
            model.predict_proba(
                test_X
            )
        )

        if len(
            probabilities
        ) != len(
            fold.test_indices
        ):
            raise WalkForwardError(
                "Model returned wrong number of probabilities."
            )

        for index, probability in zip(
            fold.test_indices,
            probabilities,
            strict=True,
        ):
            forecasts.append(
                ForecastRecord(
                    fold=fold.fold,
                    decision_time=
                        dataset.examples[
                            index
                        ].decision_time,
                    target=
                        matrix.y[
                            index
                        ],
                    probability=
                        probability,
                )
            )

    y = tuple(
        forecast.target
        for forecast in forecasts
    )

    probabilities = tuple(
        forecast.probability
        for forecast in forecasts
    )

    return EvaluationResult(
        feature_names=
            selected,
        forecasts=
            tuple(
                forecasts
            ),
        log_loss=
            log_loss(
                y,
                probabilities,
            ),
        brier_score=
            brier_score(
                y,
                probabilities,
            ),
        accuracy=
            classification_accuracy(
                y,
                probabilities,
            ),
    )
