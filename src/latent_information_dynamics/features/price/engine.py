from __future__ import annotations

import math

from datetime import datetime
from typing import Iterable

from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)

from .baseline import (
    RANGE_HORIZONS,
    RETURN_HORIZONS,
    VOLATILITY_HORIZONS,
    PriceFeatureError,
    PriceFeatureSnapshot,
    _build_index,
    _log_range,
    _log_return,
    _realized_volatility,
)


class PriceBaselineEngine:
    def __init__(
        self,
        bars: Iterable[KlineBar],
    ) -> None:
        self._index = _build_index(bars)

    @property
    def bar_count(self) -> int:
        return len(self._index)

    def snapshot(
        self,
        decision_time: datetime,
    ) -> PriceFeatureSnapshot:
        if decision_time.tzinfo is None:
            raise PriceFeatureError(
                "decision_time must be timezone-aware"
            )

        if (
            decision_time.minute != 0
            or decision_time.second != 0
            or decision_time.microsecond != 0
            or decision_time.hour % 4 != 0
        ):
            raise PriceFeatureError(
                "decision_time must be a 4-hour boundary"
            )

        features: dict[str, float] = {}

        for name, horizon in RETURN_HORIZONS.items():
            features[f"return_{name}"] = _log_return(
                self._index,
                decision_time,
                horizon,
            )

        for name, horizon in VOLATILITY_HORIZONS.items():
            features[f"rv_{name}"] = _realized_volatility(
                self._index,
                decision_time,
                horizon,
            )

        for name, horizon in RANGE_HORIZONS.items():
            features[f"range_{name}"] = _log_range(
                self._index,
                decision_time,
                horizon,
            )

        if not all(
            math.isfinite(value)
            for value in features.values()
        ):
            raise PriceFeatureError(
                "Non-finite price feature"
            )

        return PriceFeatureSnapshot(
            decision_time=decision_time,
            features=features,
        )
