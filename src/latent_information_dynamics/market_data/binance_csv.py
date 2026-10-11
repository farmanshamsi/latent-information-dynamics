from __future__ import annotations

import csv

from datetime import datetime
from pathlib import Path
from typing import Iterable

from .binance_historical import KlineBar


class BinanceCsvError(ValueError):
    pass


REQUIRED_COLUMNS = {
    "symbol",
    "interval",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
}


def read_klines_csv(
    path: str | Path,
) -> list[KlineBar]:
    source = Path(path)

    if not source.exists():
        raise BinanceCsvError(
            f"File does not exist: {source}"
        )

    bars: list[KlineBar] = []

    with source.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise BinanceCsvError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                bars.append(
                    KlineBar(
                        symbol=row["symbol"],
                        interval=row["interval"],
                        open_time=datetime.fromisoformat(
                            row["open_time"]
                        ),
                        close_time=datetime.fromisoformat(
                            row["close_time"]
                        ),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        quote_volume=float(
                            row["quote_volume"]
                        ),
                        trade_count=int(
                            row["trade_count"]
                        ),
                        taker_buy_base_volume=float(
                            row["taker_buy_base_volume"]
                        ),
                        taker_buy_quote_volume=float(
                            row["taker_buy_quote_volume"]
                        ),
                    )
                )

            except Exception as exc:
                raise BinanceCsvError(
                    f"Invalid row {source}:{row_number}"
                ) from exc

    return bars


def read_klines_directory(
    root: str | Path,
) -> list[KlineBar]:
    directory = Path(root)

    files = sorted(
        directory.glob("????-??.csv")
    )

    if not files:
        raise BinanceCsvError(
            f"No monthly CSV files in {directory}"
        )

    bars: list[KlineBar] = []

    for path in files:
        bars.extend(
            read_klines_csv(path)
        )

    return bars


def iter_klines_directory(
    root: str | Path,
) -> Iterable[KlineBar]:
    directory = Path(root)

    files = sorted(
        directory.glob("????-??.csv")
    )

    if not files:
        raise BinanceCsvError(
            f"No monthly CSV files in {directory}"
        )

    for path in files:
        yield from read_klines_csv(path)
