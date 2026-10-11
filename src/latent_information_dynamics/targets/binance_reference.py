from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from latent_information_dynamics.data.timestamps import ensure_utc
from latent_information_dynamics.market_data.binance_historical import (
    KlineBar,
)
from latent_information_dynamics.targets.binary import (
    ReferencePriceObservation,
)


FIVE_MINUTES = timedelta(minutes=5)
ONE_MILLISECOND = timedelta(milliseconds=1)

BINANCE_BOUNDARY_SOURCE = (
    "binance-spot-btcusdt-5m-boundary-close"
)


class BinanceReferenceError(ValueError):
    pass


def build_kline_index(
    bars: Iterable[KlineBar],
) -> dict[datetime, KlineBar]:
    index: dict[datetime, KlineBar] = {}

    for bar in bars:
        if bar.open_time in index:
            raise BinanceReferenceError(
                f"Duplicate kline at "
                f"{bar.open_time.isoformat()}"
            )

        index[bar.open_time] = bar

    return index


def is_regular_five_minute_bar(
    bar: KlineBar,
) -> bool:
    expected_close = (
        bar.open_time
        + FIVE_MINUTES
        - ONE_MILLISECOND
    )

    return bar.close_time == expected_close


def reference_at_boundary(
    index: dict[datetime, KlineBar],
    *,
    boundary: datetime,
    source: str = BINANCE_BOUNDARY_SOURCE,
) -> ReferencePriceObservation:
    boundary = ensure_utc(
        boundary,
        field_name="boundary",
    )

    required_open = (
        boundary
        - FIVE_MINUTES
    )

    bar = index.get(
        required_open
    )

    if bar is None:
        raise BinanceReferenceError(
            f"Missing reference bar at "
            f"{required_open.isoformat()} "
            f"for boundary "
            f"{boundary.isoformat()}."
        )

    if not is_regular_five_minute_bar(
        bar
    ):
        raise BinanceReferenceError(
            f"Irregular reference bar at "
            f"{required_open.isoformat()}."
        )

    return ReferencePriceObservation(
        source=source,
        reference_time=boundary,
        available_time=boundary,
        price=bar.close,
    )
