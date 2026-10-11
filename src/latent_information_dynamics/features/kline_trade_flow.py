from __future__ import annotations

import math

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from latent_information_dynamics.data.timestamps import (
    ensure_utc,
)
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)


class KlineTradeFlowError(RuntimeError):
    pass


FIVE_MINUTES = timedelta(minutes=5)
ONE_MILLISECOND = timedelta(milliseconds=1)


DEFAULT_KLINE_FLOW_WINDOWS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


@dataclass(frozen=True, slots=True)
class KlineTradeFlowSummary:
    label: str
    window: timedelta
    available_time: datetime
    bar_count: int
    total_quantity: float
    signed_quantity: float
    quantity_imbalance: float
    trade_count: int


def feature_names(
    windows: Mapping[str, timedelta] | None = None,
) -> tuple[str, ...]:
    supplied = (
        windows
        if windows is not None
        else DEFAULT_KLINE_FLOW_WINDOWS
    )

    return tuple(
        name
        for label in supplied
        for name in (
            f"total_quantity_{label}",
            f"signed_quantity_{label}",
            f"quantity_imbalance_{label}",
            f"trade_count_{label}",
        )
    )


def _is_regular_five_minute_bar(
    bar: KlineBar,
) -> bool:
    return (
        bar.interval == "5m"
        and bar.close_time
        == (
            bar.open_time
            + FIVE_MINUTES
            - ONE_MILLISECOND
        )
        and bar.available_time
        == (
            bar.open_time
            + FIVE_MINUTES
        )
    )


class KlineTradeFlowIndex:
    """
    Point-in-time index for price-independent executed-flow features.

    Only these KlineBar fields are used:
      * open_time / close_time / available_time
      * volume
      * taker_buy_base_volume
      * trade_count

    OHLC and quote-notional fields are deliberately unused.
    """

    def __init__(
        self,
        bars: Sequence[KlineBar],
    ) -> None:
        if not bars:
            raise KlineTradeFlowError(
                "At least one kline is required."
            )

        self._bars = tuple(
            bars
        )

        self._open_times: list[datetime] = []

        self._prefix_total = [0.0]
        self._prefix_signed = [0.0]
        self._prefix_trade_count = [0]
        self._prefix_irregular = [0]

        previous_open_time: datetime | None = None

        for bar in self._bars:
            if (
                previous_open_time is not None
                and bar.open_time
                <= previous_open_time
            ):
                raise KlineTradeFlowError(
                    "Kline open times must be strictly increasing."
                )

            if bar.interval != "5m":
                raise KlineTradeFlowError(
                    "Trade-flow index requires 5m klines."
                )

            tolerance = (
                1e-9
                * max(
                    1.0,
                    bar.volume,
                )
            )

            if (
                bar.taker_buy_base_volume
                > bar.volume + tolerance
            ):
                raise KlineTradeFlowError(
                    "Taker-buy base volume exceeds total base volume."
                )

            signed_quantity = (
                2.0
                * bar.taker_buy_base_volume
                - bar.volume
            )

            if not math.isfinite(
                signed_quantity
            ):
                raise KlineTradeFlowError(
                    "Signed quantity is non-finite."
                )

            self._open_times.append(
                bar.open_time
            )

            self._prefix_total.append(
                self._prefix_total[-1]
                + bar.volume
            )

            self._prefix_signed.append(
                self._prefix_signed[-1]
                + signed_quantity
            )

            self._prefix_trade_count.append(
                self._prefix_trade_count[-1]
                + bar.trade_count
            )

            self._prefix_irregular.append(
                self._prefix_irregular[-1]
                + int(
                    not _is_regular_five_minute_bar(
                        bar
                    )
                )
            )

            previous_open_time = (
                bar.open_time
            )

    def summarize(
        self,
        *,
        decision_time: datetime,
        label: str,
        window: timedelta,
    ) -> KlineTradeFlowSummary:
        decision_time = ensure_utc(
            decision_time,
            field_name="decision_time",
        )

        if window <= timedelta(0):
            raise ValueError(
                "window must be positive."
            )

        if (
            window
            % FIVE_MINUTES
            != timedelta(0)
        ):
            raise ValueError(
                "window must be a multiple of five minutes."
            )

        start_time = (
            decision_time
            - window
        )

        left = bisect_left(
            self._open_times,
            start_time,
        )

        right = bisect_left(
            self._open_times,
            decision_time,
        )

        expected_bars = int(
            window
            / FIVE_MINUTES
        )

        if (
            right - left
            != expected_bars
        ):
            raise KlineTradeFlowError(
                f"Incomplete {label} history "
                f"at {decision_time.isoformat()}."
            )

        if (
            left >= len(
                self._open_times
            )
            or self._open_times[left]
            != start_time
        ):
            raise KlineTradeFlowError(
                f"Missing {label} start boundary "
                f"at {decision_time.isoformat()}."
            )

        expected_last_open = (
            decision_time
            - FIVE_MINUTES
        )

        if (
            right == 0
            or self._open_times[
                right - 1
            ]
            != expected_last_open
        ):
            raise KlineTradeFlowError(
                f"Missing {label} end boundary "
                f"at {decision_time.isoformat()}."
            )

        irregular_count = (
            self._prefix_irregular[right]
            - self._prefix_irregular[left]
        )

        if irregular_count:
            raise KlineTradeFlowError(
                f"Irregular 5m bar inside {label} "
                f"window at {decision_time.isoformat()}."
            )

        total_quantity = (
            self._prefix_total[right]
            - self._prefix_total[left]
        )

        signed_quantity = (
            self._prefix_signed[right]
            - self._prefix_signed[left]
        )

        trade_count = (
            self._prefix_trade_count[right]
            - self._prefix_trade_count[left]
        )

        if total_quantity > 0.0:
            quantity_imbalance = (
                signed_quantity
                / total_quantity
            )
        else:
            quantity_imbalance = 0.0

        return KlineTradeFlowSummary(
            label=label,
            window=window,
            available_time=decision_time,
            bar_count=expected_bars,
            total_quantity=total_quantity,
            signed_quantity=signed_quantity,
            quantity_imbalance=
                quantity_imbalance,
            trade_count=trade_count,
        )

    def summaries(
        self,
        *,
        decision_time: datetime,
        windows: Mapping[str, timedelta] | None = None,
    ) -> dict[str, KlineTradeFlowSummary]:
        supplied = (
            dict(windows)
            if windows is not None
            else dict(
                DEFAULT_KLINE_FLOW_WINDOWS
            )
        )

        return {
            label: self.summarize(
                decision_time=decision_time,
                label=label,
                window=window,
            )
            for label, window
            in supplied.items()
        }
