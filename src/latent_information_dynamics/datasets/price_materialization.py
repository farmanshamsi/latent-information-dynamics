from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from latent_information_dynamics.features.price.baseline import (
    PriceFeatureError,
    PriceFeatureSnapshot,
)
from latent_information_dynamics.features.price.engine import (
    PriceBaselineEngine,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)


FOUR_HOURS = timedelta(hours=4)


class PriceMaterializationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RejectedDecision:
    decision_time: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class PriceMaterializationResult:
    snapshots: tuple[PriceFeatureSnapshot, ...]
    rejected: tuple[RejectedDecision, ...]
    theoretical_decisions: int

    @property
    def accepted_count(self) -> int:
        return len(self.snapshots)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def four_hour_boundaries(
    start: datetime,
    end: datetime,
) -> tuple[datetime, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise PriceMaterializationError(
            "start and end must be timezone-aware"
        )

    if start >= end:
        raise PriceMaterializationError(
            "start must be before end"
        )

    if (
        start.minute != 0
        or start.second != 0
        or start.microsecond != 0
        or start.hour % 4 != 0
    ):
        raise PriceMaterializationError(
            "start must be a 4-hour boundary"
        )

    boundaries = []
    cursor = start

    while cursor < end:
        boundaries.append(cursor)
        cursor += FOUR_HOURS

    return tuple(boundaries)


def materialize_price_features(
    bars: Iterable[KlineBar],
    *,
    start: datetime,
    end: datetime,
) -> PriceMaterializationResult:
    engine = PriceBaselineEngine(bars)

    decisions = four_hour_boundaries(
        start,
        end,
    )

    accepted = []
    rejected = []

    for decision_time in decisions:
        try:
            accepted.append(
                engine.snapshot(decision_time)
            )
        except PriceFeatureError as exc:
            rejected.append(
                RejectedDecision(
                    decision_time=decision_time,
                    reason=str(exc),
                )
            )

    return PriceMaterializationResult(
        snapshots=tuple(accepted),
        rejected=tuple(rejected),
        theoretical_decisions=len(decisions),
    )
