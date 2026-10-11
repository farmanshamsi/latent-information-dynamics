from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import csv
import math
import zipfile

from latent_information_dynamics.market_data.models import (
    AggressorSide,
    Trade,
)


UTC = timezone.utc


class BinanceAggTradeError(ValueError):
    pass


def _parse_bool(
    value: str,
) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise BinanceAggTradeError(
        f"Invalid boolean value: {value!r}"
    )


def _timestamp_from_binance(
    value: str,
) -> datetime:
    raw = int(
        value
    )

    # Binance Spot archives switched from
    # millisecond to microsecond timestamps
    # from 2025 onward.
    divisor = (
        1_000_000
        if raw >= 100_000_000_000_000
        else 1_000
    )

    return datetime.fromtimestamp(
        raw / divisor,
        tz=UTC,
    )


def parse_binance_aggtrade_row(
    row: list[str],
    *,
    symbol: str = "BTCUSDT",
) -> Trade:
    if len(row) != 8:
        raise BinanceAggTradeError(
            "Binance aggTrade row must "
            "contain exactly 8 columns."
        )

    aggregate_trade_id = (
        row[0].strip()
    )

    try:
        price = float(
            row[1]
        )

        quantity = float(
            row[2]
        )

        timestamp = (
            _timestamp_from_binance(
                row[5]
            )
        )

        buyer_is_maker = (
            _parse_bool(
                row[6]
            )
        )

    except (
        ValueError,
        OverflowError,
    ) as exc:
        raise BinanceAggTradeError(
            "Invalid Binance aggTrade row."
        ) from exc

    if not aggregate_trade_id:
        raise BinanceAggTradeError(
            "Aggregate trade id cannot "
            "be empty."
        )

    if (
        not math.isfinite(price)
        or price <= 0.0
    ):
        raise BinanceAggTradeError(
            "Trade price must be "
            "positive and finite."
        )

    if (
        not math.isfinite(quantity)
        or quantity <= 0.0
    ):
        raise BinanceAggTradeError(
            "Trade quantity must be "
            "positive and finite."
        )

    # If buyer is maker, the taker /
    # aggressor was the seller.
    aggressor_side = (
        AggressorSide.SELL
        if buyer_is_maker
        else AggressorSide.BUY
    )

    try:
        sequence = int(
            aggregate_trade_id
        )
    except ValueError:
        sequence = None

    return Trade(
        exchange="binance-spot",
        symbol=symbol,
        trade_id=aggregate_trade_id,
        event_time=timestamp,

        # Historical archive does not
        # contain local receive latency.
        # Event time is therefore the
        # historical availability proxy.
        receive_time=timestamp,

        price=price,
        size=quantity,
        aggressor_side=aggressor_side,
        sequence=sequence,
    )


def _is_header(
    row: list[str],
) -> bool:
    if not row:
        return False

    return (
        row[0].strip().lower()
        in {
            "agg_trade_id",
            "aggregate_trade_id",
            "id",
        }
    )


def read_binance_aggtrade_csv(
    path: Path,
    *,
    symbol: str = "BTCUSDT",
) -> list[Trade]:
    output: list[Trade] = []

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle
        )

        for row in reader:
            if not row:
                continue

            if _is_header(row):
                continue

            output.append(
                parse_binance_aggtrade_row(
                    row,
                    symbol=symbol,
                )
            )

    return output


def iter_binance_aggtrade_zip(
    path: Path,
    *,
    symbol: str = "BTCUSDT",
) -> Iterable[Trade]:
    """
    Stream normalized aggregate trades from one Binance ZIP archive.

    The archive is processed row-by-row so large BTCUSDT files do not
    need to be materialized in memory.
    """
    with zipfile.ZipFile(
        path
    ) as archive:
        members = [
            name
            for name
            in archive.namelist()
            if name.lower().endswith(
                ".csv"
            )
        ]

        if len(members) != 1:
            raise BinanceAggTradeError(
                "Expected exactly one CSV "
                "inside Binance archive."
            )

        with archive.open(
            members[0]
        ) as binary:
            lines = (
                line.decode("utf-8")
                for line in binary
            )

            reader = csv.reader(
                lines
            )

            for row in reader:
                if not row:
                    continue

                if _is_header(row):
                    continue

                yield parse_binance_aggtrade_row(
                    row,
                    symbol=symbol,
                )


def read_binance_aggtrade_zip(
    path: Path,
    *,
    symbol: str = "BTCUSDT",
) -> list[Trade]:
    return list(
        iter_binance_aggtrade_zip(
            path,
            symbol=symbol,
        )
    )


def iter_binance_aggtrade_zips(
    paths: Iterable[Path],
    *,
    symbol: str = "BTCUSDT",
) -> Iterable[Trade]:
    previous_time = None

    for path in paths:
        for trade in (
            iter_binance_aggtrade_zip(
                path,
                symbol=symbol,
            )
        ):
            if (
                previous_time is not None
                and trade.event_time
                < previous_time
            ):
                raise BinanceAggTradeError(
                    "Historical trades moved "
                    "backwards in time."
                )

            previous_time = (
                trade.event_time
            )

            yield trade
