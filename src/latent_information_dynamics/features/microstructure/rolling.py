from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from latent_information_dynamics.data.contracts import (
    NumericFeatureObservation,
)
from latent_information_dynamics.data.timestamps import (
    InformationTimestamps,
)

from .engine import (
    MicrostructureState,
)


class RollingFeatureError(RuntimeError):
    """Raised when rolling microstructure state is invalid."""


DEFAULT_WINDOWS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


@dataclass(frozen=True, slots=True)
class MicrostructureWindowSummary:
    """
    Aggregated microstructure information over one lookback horizon.

    Flow variables are accumulated.

    State variables are represented by event-weighted averages,
    extrema and/or latest values.
    """

    label: str
    window: timedelta

    available_time: datetime
    observation_count: int

    cumulative_midpoint_log_return: float
    realized_volatility: float

    ofi_sum: float
    ofi_mean: float

    relative_spread_bps_mean: float
    relative_spread_bps_max: float

    depth_imbalance_mean: float
    depth_imbalance_last: float

    top_level_imbalance_mean: float
    top_level_imbalance_last: float

    microprice_displacement_bps_mean: float
    microprice_displacement_bps_last: float

    trade_imbalance_mean: float
    trade_imbalance_last: float

    notional_imbalance_mean: float
    notional_imbalance_last: float


class MultiHorizonMicrostructureEngine:
    """
    Streaming multi-horizon aggregation of microstructure states.

    States must arrive in non-decreasing availability-time order.

    Only history required by the largest configured horizon is retained.
    """

    def __init__(
        self,
        *,
        windows: Mapping[str, timedelta] | None = None,
    ) -> None:
        supplied = (
            dict(windows)
            if windows is not None
            else dict(DEFAULT_WINDOWS)
        )

        if not supplied:
            raise ValueError(
                "At least one rolling window is required."
            )

        for label, window in supplied.items():
            if not label.strip():
                raise ValueError(
                    "Rolling-window labels cannot be empty."
                )

            if window <= timedelta(0):
                raise ValueError(
                    f"Window {label!r} must be positive."
                )

        self.windows = supplied

        self._max_window = max(
            supplied.values()
        )

        self._states: deque[
            MicrostructureState
        ] = deque()

        self._latest_time: datetime | None = None

    def update(
        self,
        state: MicrostructureState,
    ) -> dict[
        str,
        MicrostructureWindowSummary,
    ]:
        """
        Add one new point-in-time state and calculate all horizons.
        """

        if (
            self._latest_time is not None
            and state.available_time < self._latest_time
        ):
            raise RollingFeatureError(
                "Microstructure available_time moved backwards."
            )

        self._states.append(
            state
        )

        self._latest_time = state.available_time

        self._trim(
            as_of=state.available_time
        )

        return self.summaries(
            as_of=state.available_time
        )

    def _trim(
        self,
        *,
        as_of: datetime,
    ) -> None:
        cutoff = (
            as_of
            - self._max_window
        )

        while (
            self._states
            and self._states[0].available_time < cutoff
        ):
            self._states.popleft()

    def _states_for_window(
        self,
        *,
        as_of: datetime,
        window: timedelta,
    ) -> tuple[MicrostructureState, ...]:
        cutoff = (
            as_of
            - window
        )

        return tuple(
            state
            for state in self._states
            if (
                cutoff
                <= state.available_time
                <= as_of
            )
        )

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _summarize(
        self,
        *,
        label: str,
        window: timedelta,
        as_of: datetime,
    ) -> MicrostructureWindowSummary:
        states = self._states_for_window(
            as_of=as_of,
            window=window,
        )

        if not states:
            raise RollingFeatureError(
                f"No states available for window {label!r}."
            )

        returns = [
            state.midpoint_log_return
            for state in states
        ]

        ofi = [
            state.order_flow_imbalance
            for state in states
        ]

        spreads = [
            state.relative_spread_bps
            for state in states
        ]

        depth_imbalance = [
            state.depth_imbalance
            for state in states
        ]

        top_imbalance = [
            state.top_level_imbalance
            for state in states
        ]

        microprice_pressure = [
            state.microprice_displacement_bps
            for state in states
        ]

        trade_imbalance = [
            state.trade_imbalance
            for state in states
        ]

        notional_imbalance = [
            state.notional_imbalance
            for state in states
        ]

        latest = states[-1]

        return MicrostructureWindowSummary(
            label=label,
            window=window,

            available_time=as_of,
            observation_count=len(states),

            cumulative_midpoint_log_return=
                sum(returns),

            realized_volatility=
                math.sqrt(
                    sum(
                        value * value
                        for value in returns
                    )
                ),

            ofi_sum=sum(ofi),
            ofi_mean=self._mean(ofi),

            relative_spread_bps_mean=
                self._mean(spreads),

            relative_spread_bps_max=
                max(spreads),

            depth_imbalance_mean=
                self._mean(
                    depth_imbalance
                ),

            depth_imbalance_last=
                latest.depth_imbalance,

            top_level_imbalance_mean=
                self._mean(
                    top_imbalance
                ),

            top_level_imbalance_last=
                latest.top_level_imbalance,

            microprice_displacement_bps_mean=
                self._mean(
                    microprice_pressure
                ),

            microprice_displacement_bps_last=
                latest.microprice_displacement_bps,

            trade_imbalance_mean=
                self._mean(
                    trade_imbalance
                ),

            trade_imbalance_last=
                latest.trade_imbalance,

            notional_imbalance_mean=
                self._mean(
                    notional_imbalance
                ),

            notional_imbalance_last=
                latest.notional_imbalance,
        )

    def summaries(
        self,
        *,
        as_of: datetime | None = None,
    ) -> dict[
        str,
        MicrostructureWindowSummary,
    ]:
        if not self._states:
            raise RollingFeatureError(
                "No microstructure states have been observed."
            )

        if as_of is None:
            if self._latest_time is None:
                raise RollingFeatureError(
                    "Latest timestamp is unavailable."
                )

            as_of = self._latest_time

        return {
            label: self._summarize(
                label=label,
                window=window,
                as_of=as_of,
            )
            for label, window in self.windows.items()
        }
def summaries_as_feature_observations(
    summaries: Mapping[
        str,
        MicrostructureWindowSummary,
    ],
    *,
    source: str,
    entity: str,
) -> tuple[NumericFeatureObservation, ...]:
    """
    Convert multi-horizon summaries into point-in-time-safe
    model observations.

    Feature names explicitly encode their lookback horizon.
    """

    output: list[
        NumericFeatureObservation
    ] = []

    for label, summary in summaries.items():
        values = {
            f"midpoint_log_return_{label}":
                summary.cumulative_midpoint_log_return,

            f"realized_volatility_{label}":
                summary.realized_volatility,

            f"ofi_sum_{label}":
                summary.ofi_sum,

            f"ofi_mean_{label}":
                summary.ofi_mean,

            f"relative_spread_bps_mean_{label}":
                summary.relative_spread_bps_mean,

            f"relative_spread_bps_max_{label}":
                summary.relative_spread_bps_max,

            f"depth_imbalance_mean_{label}":
                summary.depth_imbalance_mean,

            f"depth_imbalance_last_{label}":
                summary.depth_imbalance_last,

            f"top_level_imbalance_mean_{label}":
                summary.top_level_imbalance_mean,

            f"top_level_imbalance_last_{label}":
                summary.top_level_imbalance_last,

            f"microprice_displacement_bps_mean_{label}":
                summary.microprice_displacement_bps_mean,

            f"microprice_displacement_bps_last_{label}":
                summary.microprice_displacement_bps_last,

            f"trade_imbalance_mean_{label}":
                summary.trade_imbalance_mean,

            f"trade_imbalance_last_{label}":
                summary.trade_imbalance_last,

            f"notional_imbalance_mean_{label}":
                summary.notional_imbalance_mean,

            f"notional_imbalance_last_{label}":
                summary.notional_imbalance_last,
        }

        timestamps = InformationTimestamps(
            event_time=
                summary.available_time,
            receive_time=
                summary.available_time,
            process_time=
                summary.available_time,
        )

        for name, value in values.items():
            output.append(
                NumericFeatureObservation(
                    name=name,
                    value=value,
                    source=source,
                    entity=entity,
                    timestamps=timestamps,
                )
            )

    return tuple(output)
