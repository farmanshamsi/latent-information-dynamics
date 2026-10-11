from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Iterable, Sequence

from latent_information_dynamics.data.feature_frame import (
    FeatureFrameRow,
)
from latent_information_dynamics.data.timestamps import (
    ensure_utc,
)
from latent_information_dynamics.targets.binary import (
    BinaryDirectionTarget,
)


class SupervisedDatasetError(ValueError):
    """Raised when a supervised research dataset is invalid."""


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    """
    One point-in-time-safe forecasting example.

    Features belong to decision_time.

    The target refers to the future resolution_time.

    label_available_time records when the realized outcome first
    became legally available for model training/evaluation.
    """

    decision_time: datetime
    resolution_time: datetime
    label_available_time: datetime

    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]

    target: int

    realized_return: float
    realized_log_return: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_time",
            ensure_utc(
                self.decision_time,
                field_name="decision_time",
            ),
        )

        object.__setattr__(
            self,
            "resolution_time",
            ensure_utc(
                self.resolution_time,
                field_name="resolution_time",
            ),
        )

        object.__setattr__(
            self,
            "label_available_time",
            ensure_utc(
                self.label_available_time,
                field_name="label_available_time",
            ),
        )

        if self.resolution_time <= self.decision_time:
            raise SupervisedDatasetError(
                "resolution_time must follow decision_time."
            )

        if self.label_available_time < self.resolution_time:
            raise SupervisedDatasetError(
                "A label cannot become available before "
                "the outcome resolution time."
            )

        if self.target not in (0, 1):
            raise SupervisedDatasetError(
                "Binary target must be 0 or 1."
            )

        if not self.feature_names:
            raise SupervisedDatasetError(
                "At least one feature is required."
            )

        if len(
            self.feature_names
        ) != len(
            self.feature_values
        ):
            raise SupervisedDatasetError(
                "feature_names and feature_values must have "
                "the same length."
            )

        if len(
            self.feature_names
        ) != len(
            set(
                self.feature_names
            )
        ):
            raise SupervisedDatasetError(
                "Feature names must be unique."
            )

        for name in self.feature_names:
            if not name.strip():
                raise SupervisedDatasetError(
                    "Feature names cannot be empty."
                )

        for value in self.feature_values:
            if not math.isfinite(
                value
            ):
                raise SupervisedDatasetError(
                    "Feature values must be finite."
                )

        if not math.isfinite(
            self.realized_return
        ):
            raise SupervisedDatasetError(
                "realized_return must be finite."
            )

        if not math.isfinite(
            self.realized_log_return
        ):
            raise SupervisedDatasetError(
                "realized_log_return must be finite."
            )

    def feature_dict(
        self,
    ) -> dict[str, float]:
        return dict(
            zip(
                self.feature_names,
                self.feature_values,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class MatrixView:
    """
    Lightweight dependency-free ML matrix representation.
    """

    feature_names: tuple[str, ...]

    X: tuple[
        tuple[float, ...],
        ...
    ]

    y: tuple[int, ...]

    decision_times: tuple[
        datetime,
        ...
    ]

    label_available_times: tuple[
        datetime,
        ...
    ]

    def __post_init__(self) -> None:
        n = len(
            self.X
        )

        if len(self.y) != n:
            raise SupervisedDatasetError(
                "X and y lengths differ."
            )

        if len(
            self.decision_times
        ) != n:
            raise SupervisedDatasetError(
                "decision_times length differs from X."
            )

        if len(
            self.label_available_times
        ) != n:
            raise SupervisedDatasetError(
                "label_available_times length differs from X."
            )

        expected_width = len(
            self.feature_names
        )

        for row in self.X:
            if len(row) != expected_width:
                raise SupervisedDatasetError(
                    "Matrix row width differs from feature schema."
                )


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    """
    Chronological immutable binary forecasting dataset.
    """

    examples: tuple[
        SupervisedExample,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.examples:
            raise SupervisedDatasetError(
                "Dataset cannot be empty."
            )

        ordered = tuple(
            sorted(
                self.examples,
                key=lambda item:
                    item.decision_time,
            )
        )

        object.__setattr__(
            self,
            "examples",
            ordered,
        )

        decision_times = [
            example.decision_time
            for example in ordered
        ]

        if len(
            decision_times
        ) != len(
            set(
                decision_times
            )
        ):
            raise SupervisedDatasetError(
                "Decision times must be unique."
            )

        schema = ordered[
            0
        ].feature_names

        for example in ordered[1:]:
            if example.feature_names != schema:
                raise SupervisedDatasetError(
                    "All examples must share the same ordered "
                    "feature schema."
                )

    @property
    def feature_names(
        self,
    ) -> tuple[str, ...]:
        return self.examples[
            0
        ].feature_names

    def available_for_training_by(
        self,
        cutoff: datetime,
    ) -> "SupervisedDataset | None":
        """
        Return only examples whose realized labels were already
        observable by the requested historical training cutoff.
        """

        cutoff = ensure_utc(
            cutoff,
            field_name="cutoff",
        )

        eligible = tuple(
            example
            for example in self.examples
            if (
                example.label_available_time
                <= cutoff
            )
        )

        if not eligible:
            return None

        return SupervisedDataset(
            examples=eligible
        )

    def matrix(
        self,
        *,
        features: Sequence[str] | None = None,
    ) -> MatrixView:
        """
        Build deterministic X/y representation.

        Feature order is explicit and preserved.
        """

        selected = (
            tuple(features)
            if features is not None
            else self.feature_names
        )

        if not selected:
            raise SupervisedDatasetError(
                "At least one matrix feature is required."
            )

        if len(
            selected
        ) != len(
            set(
                selected
            )
        ):
            raise SupervisedDatasetError(
                "Selected feature names must be unique."
            )

        unknown = [
            name
            for name in selected
            if name not in self.feature_names
        ]

        if unknown:
            raise SupervisedDatasetError(
                "Unknown feature(s): "
                + ", ".join(
                    repr(name)
                    for name in unknown
                )
            )

        X: list[
            tuple[float, ...]
        ] = []

        y: list[int] = []

        decision_times: list[
            datetime
        ] = []

        label_available_times: list[
            datetime
        ] = []

        for example in self.examples:
            feature_map = (
                example.feature_dict()
            )

            X.append(
                tuple(
                    feature_map[name]
                    for name in selected
                )
            )

            y.append(
                example.target
            )

            decision_times.append(
                example.decision_time
            )

            label_available_times.append(
                example.label_available_time
            )

        return MatrixView(
            feature_names=selected,
            X=tuple(X),
            y=tuple(y),
            decision_times=tuple(
                decision_times
            ),
            label_available_times=tuple(
                label_available_times
            ),
        )


def build_supervised_example(
    *,
    features: FeatureFrameRow,
    target: BinaryDirectionTarget,
) -> SupervisedExample:
    """
    Join one point-in-time feature vector with its future realized
    binary outcome.

    This function is intended for ex-post dataset construction.

    It does not make future target information part of the predictor.
    """

    if (
        features.prediction_time
        != target.window.decision_time
    ):
        raise SupervisedDatasetError(
            "Feature prediction_time must equal target decision_time."
        )

    if target.outcome is None:
        raise SupervisedDatasetError(
            "Unresolved/excluded target cannot enter a "
            "binary supervised dataset."
        )

    names: list[str] = []
    values: list[float] = []

    for feature in features.features:
        if feature.available_time > features.prediction_time:
            raise SupervisedDatasetError(
                f"Feature {feature.name!r} became available "
                f"after decision_time."
            )

        if not isinstance(
            feature.value,
            Real,
        ):
            raise SupervisedDatasetError(
                f"Feature {feature.name!r} is not numeric."
            )

        numeric = float(
            feature.value
        )

        if not math.isfinite(
            numeric
        ):
            raise SupervisedDatasetError(
                f"Feature {feature.name!r} is not finite."
            )

        names.append(
            feature.name
        )

        values.append(
            numeric
        )

    label_available_time = max(
        target.start_reference.available_time,
        target.end_reference.available_time,
        target.window.resolution_time,
    )

    return SupervisedExample(
        decision_time=
            target.window.decision_time,

        resolution_time=
            target.window.resolution_time,

        label_available_time=
            label_available_time,

        feature_names=
            tuple(
                names
            ),

        feature_values=
            tuple(
                values
            ),

        target=
            target.outcome,

        realized_return=
            target.arithmetic_return,

        realized_log_return=
            target.log_return,
    )


def build_supervised_dataset(
    pairs: Iterable[
        tuple[
            FeatureFrameRow,
            BinaryDirectionTarget,
        ]
    ],
) -> SupervisedDataset:
    """
    Build a chronological immutable dataset from feature/target pairs.
    """

    examples = tuple(
        build_supervised_example(
            features=features,
            target=target,
        )
        for features, target
        in pairs
    )

    return SupervisedDataset(
        examples=examples
    )
