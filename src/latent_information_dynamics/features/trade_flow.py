from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from latent_information_dynamics.market_data.models import (
    AggressorSide,
    Trade,
)


class TradeFlowError(RuntimeError):
    pass


DEFAULT_TRADE_FLOW_WINDOWS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


@dataclass(frozen=True, slots=True)
class TradeFlowSummary:
    label: str
    window: timedelta
    available_time: datetime
    aggregate_trade_count: int
    buy_aggregate_trade_count: int
    sell_aggregate_trade_count: int
    total_quantity: float
    signed_quantity: float
    quantity_imbalance: float
    count_imbalance: float


@dataclass(slots=True)
class _WindowState:
    trades: deque[Trade]
    total_quantity: float = 0.0
    signed_quantity: float = 0.0
    buy_count: int = 0
    sell_count: int = 0


class MultiHorizonTradeFlowEngine:
    """
    Streaming price-independent trade-flow aggregation.

    Features depend only on:
      * executed quantity
      * aggressor direction
      * event availability time

    Trade price and notional are deliberately unused.
    """

    def __init__(
        self,
        *,
        windows: Mapping[str, timedelta] | None = None,
    ) -> None:
        supplied = (
            dict(windows)
            if windows is not None
            else dict(DEFAULT_TRADE_FLOW_WINDOWS)
        )

        if not supplied:
            raise ValueError(
                "At least one trade-flow window is required."
            )

        for label, window in supplied.items():
            if not label.strip():
                raise ValueError(
                    "Window labels cannot be empty."
                )

            if window <= timedelta(0):
                raise ValueError(
                    f"Window {label!r} must be positive."
                )

        self.windows = supplied

        self._states = {
            label: _WindowState(
                trades=deque()
            )
            for label in supplied
        }

        self._latest_trade_time: datetime | None = None
        self._latest_as_of: datetime | None = None

    def on_trade(
        self,
        trade: Trade,
    ) -> None:
        if (
            self._latest_trade_time is not None
            and trade.receive_time
            < self._latest_trade_time
        ):
            raise TradeFlowError(
                "Trade receive_time moved backwards."
            )

        if trade.aggressor_side is AggressorSide.UNKNOWN:
            raise TradeFlowError(
                "Aggressor side is required for trade-flow features."
            )

        signed_quantity = trade.signed_size

        for state in self._states.values():
            state.trades.append(
                trade
            )

            state.total_quantity += (
                trade.size
            )

            state.signed_quantity += (
                signed_quantity
            )

            if (
                trade.aggressor_side
                is AggressorSide.BUY
            ):
                state.buy_count += 1
            else:
                state.sell_count += 1

        self._latest_trade_time = (
            trade.receive_time
        )

    @staticmethod
    def _remove_trade(
        state: _WindowState,
        trade: Trade,
    ) -> None:
        state.total_quantity -= (
            trade.size
        )

        state.signed_quantity -= (
            trade.signed_size
        )

        if (
            trade.aggressor_side
            is AggressorSide.BUY
        ):
            state.buy_count -= 1
        else:
            state.sell_count -= 1

    def _trim(
        self,
        *,
        label: str,
        as_of: datetime,
    ) -> None:
        state = self._states[
            label
        ]

        cutoff = (
            as_of
            - self.windows[label]
        )

        while (
            state.trades
            and state.trades[0].receive_time
            < cutoff
        ):
            removed = (
                state.trades.popleft()
            )

            self._remove_trade(
                state,
                removed,
            )

    def summaries(
        self,
        *,
        as_of: datetime,
    ) -> dict[str, TradeFlowSummary]:
        if (
            self._latest_as_of is not None
            and as_of < self._latest_as_of
        ):
            raise TradeFlowError(
                "Trade-flow as_of moved backwards."
            )

        if (
            self._latest_trade_time is not None
            and self._latest_trade_time > as_of
        ):
            raise TradeFlowError(
                "Engine contains trade information "
                "from after requested as_of."
            )

        output = {}

        for label, window in self.windows.items():
            self._trim(
                label=label,
                as_of=as_of,
            )

            state = self._states[
                label
            ]

            count = (
                state.buy_count
                + state.sell_count
            )

            if state.total_quantity > 0.0:
                quantity_imbalance = (
                    state.signed_quantity
                    / state.total_quantity
                )
            else:
                quantity_imbalance = 0.0

            if count > 0:
                count_imbalance = (
                    state.buy_count
                    - state.sell_count
                ) / count
            else:
                count_imbalance = 0.0

            values = (
                state.total_quantity,
                state.signed_quantity,
                quantity_imbalance,
                count_imbalance,
            )

            if any(
                not math.isfinite(value)
                for value in values
            ):
                raise TradeFlowError(
                    f"Non-finite trade-flow state for {label!r}."
                )

            output[label] = (
                TradeFlowSummary(
                    label=label,
                    window=window,
                    available_time=as_of,
                    aggregate_trade_count=count,
                    buy_aggregate_trade_count=
                        state.buy_count,
                    sell_aggregate_trade_count=
                        state.sell_count,
                    total_quantity=
                        state.total_quantity,
                    signed_quantity=
                        state.signed_quantity,
                    quantity_imbalance=
                        quantity_imbalance,
                    count_imbalance=
                        count_imbalance,
                )
            )

        self._latest_as_of = as_of

        return output
