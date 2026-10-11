from __future__ import annotations

import time

from datetime import datetime, timezone
from pathlib import Path

from latent_information_dynamics.market_data.binance_historical import (
    BinanceSpotKlineClient,
    write_klines_csv,
)


UTC = timezone.utc

START = datetime(
    2020, 1, 1,
    tzinfo=UTC,
)

END = datetime(
    2026, 1, 1,
    tzinfo=UTC,
)

OUTPUT_ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)


def next_month(
    value: datetime,
) -> datetime:
    if value.month == 12:
        return datetime(
            value.year + 1,
            1,
            1,
            tzinfo=UTC,
        )

    return datetime(
        value.year,
        value.month + 1,
        1,
        tzinfo=UTC,
    )


def main() -> None:
    client = BinanceSpotKlineClient()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cursor = START

    total_bars = 0

    while cursor < END:
        month_end = min(
            next_month(cursor),
            END,
        )

        output = (
            OUTPUT_ROOT
            / f"{cursor:%Y-%m}.csv"
        )

        if output.exists():
            print(
                f"SKIP {cursor:%Y-%m}: "
                f"{output} already exists"
            )

            cursor = month_end
            continue

        print(
            f"FETCH {cursor:%Y-%m}: "
            f"{cursor.isoformat()} "
            f"→ {month_end.isoformat()}"
        )

        bars = client.fetch_range(
            symbol="BTCUSDT",
            interval="5m",
            start_time=cursor,
            end_time=month_end,
        )

        if not bars:
            raise RuntimeError(
                f"No bars returned for "
                f"{cursor:%Y-%m}"
            )

        write_klines_csv(
            bars,
            output,
        )

        total_bars += len(
            bars
        )

        print(
            f"  {len(bars):,} bars "
            f"→ {output}"
        )

        cursor = month_end

        # Gentle pause between monthly jobs.
        time.sleep(0.25)

    print()
    print(
        f"Download complete. "
        f"New bars written: "
        f"{total_bars:,}"
    )


if __name__ == "__main__":
    main()
