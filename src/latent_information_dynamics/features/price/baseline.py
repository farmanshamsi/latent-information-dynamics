from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from latent_information_dynamics.market_data.binance_historical import KlineBar


FIVE_MINUTES = timedelta(minutes=5)

RETURN_HORIZONS = {
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "12h": timedelta(hours=12),
    "24h": timedelta(hours=24),
}

VOLATILITY_HORIZONS = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "24h": timedelta(hours=24),
}

RANGE_HORIZONS = {
    "4h": timedelta(hours=4),
    "24h": timedelta(hours=24),
}


class PriceFeatureError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PriceFeatureSnapshot:
    decision_time: datetime
    features: dict[str, float]


def _is_normal_bar(bar: KlineBar) -> bool:
    expected_close = (
        bar.open_time
        + FIVE_MINUTES
        - timedelta(milliseconds=1)
    )
    return bar.close_time == expected_close


def _build_index(
    bars: Iterable[KlineBar],
) -> dict[datetime, KlineBar]:
    index: dict[datetime, KlineBar] = {}

    for bar in bars:
        if bar.open_time in index:
            raise PriceFeatureError(
                f"Duplicate bar: {bar.open_time}"
            )

        index[bar.open_time] = bar

    return index


def _required_bars(
    index: dict[datetime, KlineBar],
    start: datetime,
    end: datetime,
) -> list[KlineBar]:
    bars: list[KlineBar] = []

    cursor = start

    while cursor < end:
        bar = index.get(cursor)

        if bar is None:
            raise PriceFeatureError(
                f"Missing bar at {cursor.isoformat()}"
            )

        if not _is_normal_bar(bar):
            raise PriceFeatureError(
                f"Irregular bar at {cursor.isoformat()}"
            )

        bars.append(bar)
        cursor += FIVE_MINUTES

    return bars


def _price_at_boundary(
    index: dict[datetime, KlineBar],
    boundary: datetime,
) -> float:
    bar_time = boundary - FIVE_MINUTES

    bars = _required_bars(
        index,
        bar_time,
        boundary,
    )

    return bars[-1].close


def _log_return(
    index: dict[datetime, KlineBar],
    decision_time: datetime,
    horizon: timedelta,
) -> float:
    current = _price_at_boundary(
        index,
        decision_time,
    )

    previous = _price_at_boundary(
        index,
        decision_time - horizon,
    )

    return math.log(
        current / previous
    )


def _realized_volatility(
    index: dict[datetime, KlineBar],
    decision_time: datetime,
    horizon: timedelta,
) -> float:
    start = (
        decision_time
        - horizon
        - FIVE_MINUTES
    )

    bars = _required_bars(
        index,
        start,
        decision_time,
    )

    returns = [
        math.log(
            bars[i].close
            / bars[i - 1].close
        )
        for i in range(
            1,
            len(bars),
        )
    ]

    return math.sqrt(
        sum(
            value * value
            for value in returns
        )
    )


def _log_range(
    index: dict[datetime, KlineBar],
    decision_time: datetime,
    horizon: timedelta,
) -> float:
    bars = _required_bars(
        index,
        decision_time - horizon,
        decision_time,
    )

    highest = max(
        bar.high
        for bar in bars
    )

    lowest = min(
        bar.low
        for bar in bars
    )

    return math.log(
        highest / lowest
    )


def build_price_feature_snapshot(
    bars: Iterable[KlineBar],
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

    index = _build_index(
        bars
    )

    features: dict[str, float] = {}

    for name, horizon in RETURN_HORIZONS.items():
        features[f"return_{name}"] = _log_return(
            index,
            decision_time,
            horizon,
        )

    for name, horizon in VOLATILITY_HORIZONS.items():
        features[f"rv_{name}"] = _realized_volatility(
            index,
            decision_time,
            horizon,
        )

    for name, horizon in RANGE_HORIZONS.items():
        features[f"range_{name}"] = _log_range(
            index,
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
