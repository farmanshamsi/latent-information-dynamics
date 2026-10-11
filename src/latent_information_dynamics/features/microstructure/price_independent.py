from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from latent_information_dynamics.data.timestamps import (
    ensure_utc,
)

from .rolling import (
    MicrostructureWindowSummary,
)


class PriceIndependentFeatureError(ValueError):
    pass


DEFAULT_WINDOW_LABELS = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
)


STRICT_COMPONENTS = (
    "depth_imbalance_mean",
    "depth_imbalance_last",
    "top_level_imbalance_mean",
    "top_level_imbalance_last",
    "trade_imbalance_mean",
    "trade_imbalance_last",
)


@dataclass(frozen=True, slots=True)
class PriceIndependentMicrostructureVector:
    available_time: datetime
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_time",
            ensure_utc(
                self.available_time,
                field_name="available_time",
            ),
        )

        if not self.feature_names:
            raise PriceIndependentFeatureError(
                "Feature vector cannot be empty."
            )

        if (
            len(self.feature_names)
            != len(self.feature_values)
        ):
            raise PriceIndependentFeatureError(
                "Feature names and values differ in length."
            )

        if (
            len(self.feature_names)
            != len(set(self.feature_names))
        ):
            raise PriceIndependentFeatureError(
                "Feature names must be unique."
            )

        if any(
            not math.isfinite(value)
            for value in self.feature_values
        ):
            raise PriceIndependentFeatureError(
                "All feature values must be finite."
            )

    def as_dict(
        self,
    ) -> dict[str, float]:
        return dict(
            zip(
                self.feature_names,
                self.feature_values,
                strict=True,
            )
        )


def strict_feature_names(
    window_labels: Sequence[str] = DEFAULT_WINDOW_LABELS,
) -> tuple[str, ...]:
    labels = tuple(
        window_labels
    )

    if not labels:
        raise PriceIndependentFeatureError(
            "At least one horizon is required."
        )

    if len(labels) != len(set(labels)):
        raise PriceIndependentFeatureError(
            "Horizon labels must be unique."
        )

    return tuple(
        f"{component}_{label}"
        for label in labels
        for component in STRICT_COMPONENTS
    )


def extract_strict_price_independent_features(
    summaries: Mapping[
        str,
        MicrostructureWindowSummary,
    ],
    *,
    window_labels: Sequence[str] = DEFAULT_WINDOW_LABELS,
) -> PriceIndependentMicrostructureVector:
    labels = tuple(
        window_labels
    )

    names = strict_feature_names(
        labels
    )

    missing = [
        label
        for label in labels
        if label not in summaries
    ]

    if missing:
        raise PriceIndependentFeatureError(
            "Missing microstructure horizon(s): "
            + ", ".join(missing)
        )

    available_times = {
        summaries[label].available_time
        for label in labels
    }

    if len(available_times) != 1:
        raise PriceIndependentFeatureError(
            "All horizon summaries must share "
            "one point-in-time availability timestamp."
        )

    values: list[float] = []

    for label in labels:
        summary = summaries[label]

        if summary.label != label:
            raise PriceIndependentFeatureError(
                f"Summary label mismatch for {label!r}."
            )

        for component in STRICT_COMPONENTS:
            value = float(
                getattr(
                    summary,
                    component,
                )
            )

            if not math.isfinite(value):
                raise PriceIndependentFeatureError(
                    f"Non-finite feature "
                    f"{component}_{label}."
                )

            values.append(
                value
            )

    return PriceIndependentMicrostructureVector(
        available_time=
            next(iter(available_times)),
        feature_names=names,
        feature_values=tuple(values),
    )
