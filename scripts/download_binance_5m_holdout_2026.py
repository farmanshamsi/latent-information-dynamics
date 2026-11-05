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
    2026, 1, 1,
    tzinfo=UTC,
)

END = datetime(
    2026, 7, 1,
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

    print(
        "BTCUSDT 5m HOLDOUT ACQUISITION"
    )
    print(
        "=============================="
    )
    print(
        "Start:",
        START.isoformat(),
    )
    print(
        "End  :",
        END.isoformat(),
    )
    print()

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

        if (
            bars[0].open_time
            != cursor
        ):
            raise RuntimeError(
                "First returned bar does not "
                "match requested month start."
            )

        if any(
            bar.open_time
            < cursor
            or bar.open_time
            >= month_end
            for bar in bars
        ):
            raise RuntimeError(
                "Returned bar lies outside "
                "requested month."
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

        time.sleep(
            0.25
        )

    print()
    print(
        "Holdout download complete."
    )
    print(
        "New bars written:",
        f"{total_bars:,}",
    )


if __name__ == "__main__":
    main()
