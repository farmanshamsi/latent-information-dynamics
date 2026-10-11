from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)
from latent_information_dynamics.targets.binary import (
    BinaryDirectionTarget,
    DecisionWindow,
    TieRule,
    build_binary_direction_target,
)
from latent_information_dynamics.targets.binance_reference import (
    BinanceReferenceError,
    build_kline_index,
    reference_at_boundary,
)


FOUR_HOURS = timedelta(hours=4)


@dataclass(frozen=True, slots=True)
class RejectedTarget:
    decision_time: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class TargetMaterializationResult:
    targets: tuple[
        BinaryDirectionTarget,
        ...
    ]
    rejected: tuple[
        RejectedTarget,
        ...
    ]
    requested_decisions: int

    @property
    def accepted_count(self) -> int:
        return len(self.targets)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def materialize_binary_targets(
    bars: Iterable[KlineBar],
    *,
    decision_times: Iterable[datetime],
    horizon: timedelta = FOUR_HOURS,
    tie_rule: TieRule = TieRule.UP,
) -> TargetMaterializationResult:
    index = build_kline_index(
        bars
    )

    decisions = tuple(
        decision_times
    )

    accepted: list[
        BinaryDirectionTarget
    ] = []

    rejected: list[
        RejectedTarget
    ] = []

    for decision_time in decisions:
        resolution_time = (
            decision_time
            + horizon
        )

        try:
            start_reference = (
                reference_at_boundary(
                    index,
                    boundary=decision_time,
                )
            )

            end_reference = (
                reference_at_boundary(
                    index,
                    boundary=resolution_time,
                )
            )

            window = DecisionWindow(
                decision_time=decision_time,
                resolution_time=resolution_time,
            )

            target = (
                build_binary_direction_target(
                    window=window,
                    start_reference=start_reference,
                    end_reference=end_reference,
                    tie_rule=tie_rule,
                )
            )

            accepted.append(
                target
            )

        except BinanceReferenceError as exc:
            rejected.append(
                RejectedTarget(
                    decision_time=decision_time,
                    reason=str(exc),
                )
            )

    return TargetMaterializationResult(
        targets=tuple(accepted),
        rejected=tuple(rejected),
        requested_decisions=len(decisions),
    )
