from __future__ import annotations

import csv
import json
import math

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from latent_information_dynamics.data.timestamps import ensure_utc


UTC = timezone.utc


class HistoricalMarketDataError(RuntimeError):
    """Raised when historical market data cannot be validated."""


@dataclass(frozen=True, slots=True)
class KlineBar:
    """
    Canonical completed OHLCV bar.

    `available_time` is the first instant after the completed bar.
    """

    symbol: str
    interval: str

    open_time: datetime
    close_time: datetime

    open: float
    high: float
    low: float
    close: float

    volume: float
    quote_volume: float

    trade_count: int

    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise HistoricalMarketDataError(
                "symbol cannot be empty."
            )

        if not self.interval.strip():
            raise HistoricalMarketDataError(
                "interval cannot be empty."
            )

        object.__setattr__(
            self,
            "open_time",
            ensure_utc(
                self.open_time,
                field_name="open_time",
            ),
        )

        object.__setattr__(
            self,
            "close_time",
            ensure_utc(
                self.close_time,
                field_name="close_time",
            ),
        )

        if self.close_time <= self.open_time:
            raise HistoricalMarketDataError(
                "close_time must follow open_time."
            )

        prices = (
            self.open,
            self.high,
            self.low,
            self.close,
        )

        if any(
            not math.isfinite(value)
            or value <= 0
            for value in prices
        ):
            raise HistoricalMarketDataError(
                "OHLC prices must be finite and positive."
            )

        if self.low > min(
            self.open,
            self.close,
        ):
            raise HistoricalMarketDataError(
                "low cannot exceed open or close."
            )

        if self.high < max(
            self.open,
            self.close,
        ):
            raise HistoricalMarketDataError(
                "high cannot be below open or close."
            )

        if self.high < self.low:
            raise HistoricalMarketDataError(
                "high cannot be below low."
            )

        volumes = (
            self.volume,
            self.quote_volume,
            self.taker_buy_base_volume,
            self.taker_buy_quote_volume,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in volumes
        ):
            raise HistoricalMarketDataError(
                "Volumes must be finite and non-negative."
            )

        if self.trade_count < 0:
            raise HistoricalMarketDataError(
                "trade_count cannot be negative."
            )

    @property
    def available_time(self) -> datetime:
        return (
            self.close_time
            + timedelta(
                milliseconds=1
            )
        )

    @property
    def log_return(self) -> float:
        return math.log(
            self.close
            / self.open
        )

    @property
    def taker_buy_fraction(self) -> float:
        if self.volume == 0:
            return 0.0

        return (
            self.taker_buy_base_volume
            / self.volume
        )


def _datetime_from_ms(
    value: int,
) -> datetime:
    return datetime.fromtimestamp(
        value / 1000.0,
        tz=UTC,
    )


def _datetime_to_ms(
    value: datetime,
) -> int:
    value = ensure_utc(
        value,
        field_name="timestamp",
    )

    return int(
        value.timestamp()
        * 1000
    )


def parse_binance_kline(
    row: list,
    *,
    symbol: str,
    interval: str,
) -> KlineBar:
    """
    Convert one Binance REST kline response row into canonical form.
    """

    if len(row) < 11:
        raise HistoricalMarketDataError(
            "Binance kline row is incomplete."
        )

    return KlineBar(
        symbol=symbol,
        interval=interval,

        open_time=
            _datetime_from_ms(
                int(row[0])
            ),

        open=float(
            row[1]
        ),

        high=float(
            row[2]
        ),

        low=float(
            row[3]
        ),

        close=float(
            row[4]
        ),

        volume=float(
            row[5]
        ),

        close_time=
            _datetime_from_ms(
                int(row[6])
            ),

        quote_volume=float(
            row[7]
        ),

        trade_count=int(
            row[8]
        ),

        taker_buy_base_volume=float(
            row[9]
        ),

        taker_buy_quote_volume=float(
            row[10]
        ),
    )


class BinanceSpotKlineClient:
    """
    Public historical Binance Spot market-data client.

    No account credentials are required for public kline data.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://data-api.binance.vision",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout_seconds = (
            timeout_seconds
        )

    def _request_json(
        self,
        path: str,
        parameters: dict[str, object],
    ) -> object:
        query = urlencode(
            parameters
        )

        url = (
            f"{self.base_url}"
            f"{path}"
            f"?{query}"
        )

        request = Request(
            url,
            headers={
                "User-Agent":
                    "latent-information-dynamics/1.0"
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                payload = response.read()

        except Exception as exc:
            raise HistoricalMarketDataError(
                f"Binance request failed: {url}"
            ) from exc

        try:
            return json.loads(
                payload.decode(
                    "utf-8"
                )
            )

        except json.JSONDecodeError as exc:
            raise HistoricalMarketDataError(
                "Binance returned invalid JSON."
            ) from exc

    def fetch_page(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> tuple[KlineBar, ...]:
        if limit <= 0 or limit > 1000:
            raise ValueError(
                "limit must lie between 1 and 1000."
            )

        parameters: dict[
            str,
            object,
        ] = {
            "symbol":
                symbol.upper(),
            "interval":
                interval,
            "startTime":
                _datetime_to_ms(
                    start_time
                ),
            "limit":
                limit,
        }

        if end_time is not None:
            parameters[
                "endTime"
            ] = _datetime_to_ms(
                end_time
            )

        payload = self._request_json(
            "/api/v3/klines",
            parameters,
        )

        if not isinstance(
            payload,
            list,
        ):
            raise HistoricalMarketDataError(
                "Unexpected Binance response structure."
            )

        bars = tuple(
            parse_binance_kline(
                row,
                symbol=
                    symbol.upper(),
                interval=
                    interval,
            )
            for row in payload
        )

        return bars

    def fetch_range(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> tuple[KlineBar, ...]:
        """
        Paginate historical klines over a requested UTC interval.
        """

        start_time = ensure_utc(
            start_time,
            field_name="start_time",
        )

        end_time = ensure_utc(
            end_time,
            field_name="end_time",
        )

        if end_time <= start_time:
            raise ValueError(
                "end_time must follow start_time."
            )

        cursor = start_time

        output: list[
            KlineBar
        ] = []

        previous_open_time: datetime | None = None

        while cursor < end_time:
            page = self.fetch_page(
                symbol=symbol,
                interval=interval,
                start_time=cursor,
                end_time=end_time,
                limit=limit,
            )

            if not page:
                break

            for bar in page:
                if bar.open_time >= end_time:
                    continue

                if (
                    output
                    and bar.open_time
                    <= output[-1].open_time
                ):
                    continue

                output.append(
                    bar
                )

            last_open_time = (
                page[-1].open_time
            )

            if (
                previous_open_time is not None
                and last_open_time
                <= previous_open_time
            ):
                raise HistoricalMarketDataError(
                    "Binance pagination failed to advance."
                )

            previous_open_time = (
                last_open_time
            )

            cursor = (
                last_open_time
                + timedelta(
                    milliseconds=1
                )
            )

            if len(page) < limit:
                break

        return tuple(
            output
        )


CSV_FIELDS = (
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
)


def write_klines_csv(
    bars: Iterable[KlineBar],
    path: str | Path,
) -> Path:
    """
    Persist canonical bars in deterministic chronological CSV format.
    """

    output = Path(
        path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = sorted(
        bars,
        key=lambda item:
            item.open_time,
    )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for bar in ordered:
            writer.writerow(
                {
                    "symbol":
                        bar.symbol,
                    "interval":
                        bar.interval,
                    "open_time":
                        bar.open_time.isoformat(),
                    "close_time":
                        bar.close_time.isoformat(),
                    "open":
                        repr(bar.open),
                    "high":
                        repr(bar.high),
                    "low":
                        repr(bar.low),
                    "close":
                        repr(bar.close),
                    "volume":
                        repr(bar.volume),
                    "quote_volume":
                        repr(
                            bar.quote_volume
                        ),
                    "trade_count":
                        bar.trade_count,
                    "taker_buy_base_volume":
                        repr(
                            bar.taker_buy_base_volume
                        ),
                    "taker_buy_quote_volume":
                        repr(
                            bar.taker_buy_quote_volume
                        ),
                }
            )

    return output
