from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from latent_information_dynamics.data.contracts import (
    NumericFeatureObservation,
)
from latent_information_dynamics.data.timestamps import (
    InformationTimestamps,
)
from latent_information_dynamics.market_data.book import (
    BookMetrics,
    L2OrderBook,
)
from latent_information_dynamics.market_data.models import (
    Trade,
)


class MicrostructureFeatureError(RuntimeError):
    """Raised when streaming microstructure state is invalid."""


@dataclass(frozen=True, slots=True)
class MicrostructureState:
    """
    Point-in-time microstructure feature state.

    All values are calculated exclusively from information whose
    receive time is no later than `available_time`.
    """

    event_time: datetime
    available_time: datetime

    midpoint: float
    midpoint_log_return: float

    spread: float
    relative_spread_bps: float

    top_level_imbalance: float
    depth_imbalance: float

    microprice: float
    microprice_displacement_bps: float

    order_flow_imbalance: float

    trade_count: int
    traded_size: float
    signed_traded_size: float
    trade_imbalance: float

    traded_notional: float
    signed_traded_notional: float
    notional_imbalance: float


def top_of_book_ofi(
    previous: BookMetrics,
    current: BookMetrics,
) -> float:
    """
    Event-based top-of-book Order Flow Imbalance.

    Based on changes in best bid/ask prices and displayed queue sizes.

    Positive OFI represents net buy-side pressure.
    Negative OFI represents net sell-side pressure.
    """

    bid_component = 0.0

    if current.best_bid >= previous.best_bid:
        bid_component += current.best_bid_size

    if current.best_bid <= previous.best_bid:
        bid_component -= previous.best_bid_size

    ask_component = 0.0

    if current.best_ask <= previous.best_ask:
        ask_component -= current.best_ask_size

    if current.best_ask >= previous.best_ask:
        ask_component += previous.best_ask_size

    return bid_component - ask_component


class MicrostructureFeatureEngine:
    """
    Incremental feature engine suitable for historical replay
    and live streaming market data.

    Trades are aggregated by receive_time rather than event_time
    to preserve point-in-time availability.
    """

    def __init__(
        self,
        *,
        depth_levels: int = 5,
        trade_window: timedelta = timedelta(minutes=5),
    ) -> None:
        if depth_levels <= 0:
            raise ValueError(
                "depth_levels must be positive."
            )

        if trade_window <= timedelta(0):
            raise ValueError(
                "trade_window must be positive."
            )

        self.depth_levels = depth_levels
        self.trade_window = trade_window

        self._trades: deque[Trade] = deque()

        self._previous_book_metrics: BookMetrics | None = None
        self._previous_midpoint: float | None = None
        self._previous_book_time: datetime | None = None
        self._latest_trade_time: datetime | None = None

    def on_trade(
        self,
        trade: Trade,
    ) -> None:
        """
        Add one legally received trade to rolling state.
        """

        if (
            self._latest_trade_time is not None
            and trade.receive_time < self._latest_trade_time
        ):
            raise MicrostructureFeatureError(
                "Trade receive_time moved backwards."
            )

        self._trades.append(
            trade
        )

        self._latest_trade_time = trade.receive_time

    def _trim_trades(
        self,
        *,
        as_of: datetime,
    ) -> None:
        cutoff = (
            as_of
            - self.trade_window
        )

        while (
            self._trades
            and self._trades[0].receive_time < cutoff
        ):
            self._trades.popleft()

    def _eligible_trades(
        self,
        *,
        as_of: datetime,
    ) -> tuple[Trade, ...]:
        self._trim_trades(
            as_of=as_of
        )

        return tuple(
            trade
            for trade in self._trades
            if trade.receive_time <= as_of
        )

    def snapshot(
        self,
        book: L2OrderBook,
    ) -> MicrostructureState:
        """
        Construct current point-in-time microstructure state.
        """

        book_snapshot = book.snapshot()

        event_time = book_snapshot.event_time
        available_time = book_snapshot.receive_time

        if (
            self._previous_book_time is not None
            and available_time < self._previous_book_time
        ):
            raise MicrostructureFeatureError(
                "Book receive_time moved backwards."
            )

        metrics = book.metrics()

        midpoint = metrics.midpoint

        if self._previous_midpoint is None:
            midpoint_log_return = 0.0
        else:
            midpoint_log_return = math.log(
                midpoint
                / self._previous_midpoint
            )

        relative_spread_bps = (
            metrics.spread
            / midpoint
            * 10_000.0
        )

        microprice_displacement_bps = (
            (
                metrics.microprice
                - midpoint
            )
            / midpoint
            * 10_000.0
        )

        if self._previous_book_metrics is None:
            order_flow_imbalance = 0.0
        else:
            order_flow_imbalance = top_of_book_ofi(
                self._previous_book_metrics,
                metrics,
            )

        depth_imbalance = book.depth_imbalance(
            levels=self.depth_levels
        )

        trades = self._eligible_trades(
            as_of=available_time
        )

        traded_size = sum(
            trade.size
            for trade in trades
        )

        signed_traded_size = sum(
            trade.signed_size
            for trade in trades
        )

        if traded_size > 0:
            trade_imbalance = (
                signed_traded_size
                / traded_size
            )
        else:
            trade_imbalance = 0.0

        traded_notional = sum(
            trade.notional
            for trade in trades
        )

        signed_traded_notional = sum(
            trade.signed_notional
            for trade in trades
        )

        if traded_notional > 0:
            notional_imbalance = (
                signed_traded_notional
                / traded_notional
            )
        else:
            notional_imbalance = 0.0

        state = MicrostructureState(
            event_time=event_time,
            available_time=available_time,

            midpoint=midpoint,
            midpoint_log_return=midpoint_log_return,

            spread=metrics.spread,
            relative_spread_bps=relative_spread_bps,

            top_level_imbalance=
                metrics.top_level_imbalance,

            depth_imbalance=
                depth_imbalance,

            microprice=
                metrics.microprice,

            microprice_displacement_bps=
                microprice_displacement_bps,

            order_flow_imbalance=
                order_flow_imbalance,

            trade_count=len(
                trades
            ),

            traded_size=
                traded_size,

            signed_traded_size=
                signed_traded_size,

            trade_imbalance=
                trade_imbalance,

            traded_notional=
                traded_notional,

            signed_traded_notional=
                signed_traded_notional,

            notional_imbalance=
                notional_imbalance,
        )

        self._previous_book_metrics = metrics
        self._previous_midpoint = midpoint
        self._previous_book_time = available_time

        return state
def as_feature_observations(
    state: MicrostructureState,
    *,
    source: str,
    entity: str,
) -> tuple[NumericFeatureObservation, ...]:
    """
    Convert streaming microstructure state into point-in-time-safe
    observations consumable by the multimodal feature pipeline.
    """

    timestamps = InformationTimestamps(
        event_time=state.event_time,
        receive_time=state.available_time,
        process_time=state.available_time,
    )

    values = {
        "midpoint":
            state.midpoint,

        "midpoint_log_return":
            state.midpoint_log_return,

        "spread":
            state.spread,

        "relative_spread_bps":
            state.relative_spread_bps,

        "top_level_imbalance":
            state.top_level_imbalance,

        "depth_imbalance":
            state.depth_imbalance,

        "microprice":
            state.microprice,

        "microprice_displacement_bps":
            state.microprice_displacement_bps,

        "order_flow_imbalance":
            state.order_flow_imbalance,

        "trade_count":
            float(
                state.trade_count
            ),

        "traded_size":
            state.traded_size,

        "signed_traded_size":
            state.signed_traded_size,

        "trade_imbalance":
            state.trade_imbalance,

        "traded_notional":
            state.traded_notional,

        "signed_traded_notional":
            state.signed_traded_notional,

        "notional_imbalance":
            state.notional_imbalance,
    }

    return tuple(
        NumericFeatureObservation(
            name=name,
            value=value,
            source=source,
            entity=entity,
            timestamps=timestamps,
        )
        for name, value in values.items()
    )
