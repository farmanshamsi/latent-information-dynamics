from __future__ import annotations

import csv

from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc

ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

START = datetime(
    2020, 1, 1,
    tzinfo=UTC,
)

END = datetime(
    2026, 1, 1,
    tzinfo=UTC,
)

INTERVAL = timedelta(
    minutes=5
)

BAR_CLOSE_OFFSET = (
    INTERVAL
    - timedelta(
        milliseconds=1
    )
)


def expected_month_names() -> list[str]:
    names = []

    year = 2020
    month = 1

    while (year, month) < (2026, 1):
        names.append(
            f"{year:04d}-{month:02d}.csv"
        )

        month += 1

        if month == 13:
            year += 1
            month = 1

    return names


def main() -> None:
    files = sorted(
        ROOT.glob("????-??.csv")
    )

    expected_files = expected_month_names()

    actual_files = [
        path.name
        for path in files
    ]

    missing_files = sorted(
        set(expected_files)
        - set(actual_files)
    )

    unexpected_files = sorted(
        set(actual_files)
        - set(expected_files)
    )

    total_rows = 0
    gaps = 0
    duplicates = 0
    invalid_rows = 0
    irregular_rows = 0
    out_of_order = 0

    first_open = None
    last_open = None
    previous_open = None

    for path in files:
        with path.open(
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                total_rows += 1

                try:
                    open_time = (
                        datetime.fromisoformat(
                            row["open_time"]
                        )
                    )

                    close_time = (
                        datetime.fromisoformat(
                            row["close_time"]
                        )
                    )

                except Exception:
                    invalid_rows += 1

                    print(
                        f"BAD TIMESTAMP "
                        f"{path}:{row_number}"
                    )

                    continue

                if first_open is None:
                    first_open = open_time

                last_open = open_time

                valid = True

                if row["symbol"] != "BTCUSDT":
                    valid = False

                if row["interval"] != "5m":
                    valid = False

                if open_time.tzinfo is None:
                    valid = False

                nominal_close = (
                    open_time
                    + BAR_CLOSE_OFFSET
                )

                slot_end = (
                    open_time
                    + INTERVAL
                )

                if close_time != nominal_close:
                    if (
                        open_time
                        < close_time
                        < slot_end
                    ):
                        irregular_rows += 1

                        print(
                            f"IRREGULAR CLOSE "
                            f"{path}:{row_number} "
                            f"{open_time.isoformat()} "
                            f"→ {close_time.isoformat()}"
                        )
                    else:
                        valid = False

                if (
                    open_time.second != 0
                    or open_time.microsecond != 0
                    or open_time.minute % 5 != 0
                ):
                    valid = False

                if not (
                    START
                    <= open_time
                    < END
                ):
                    valid = False

                if (
                    path.stem
                    != open_time.strftime(
                        "%Y-%m"
                    )
                ):
                    valid = False

                if not valid:
                    invalid_rows += 1

                    print(
                        f"INVALID ROW "
                        f"{path}:{row_number} "
                        f"{open_time.isoformat()}"
                    )

                if previous_open is not None:
                    expected = (
                        previous_open
                        + INTERVAL
                    )

                    if open_time == previous_open:
                        duplicates += 1

                    elif open_time < expected:
                        out_of_order += 1

                    elif open_time > expected:
                        missing = int(
                            (
                                open_time
                                - expected
                            ).total_seconds()
                            // INTERVAL.total_seconds()
                        )

                        gaps += missing

                        print(
                            f"GAP: expected "
                            f"{expected.isoformat()}, "
                            f"received "
                            f"{open_time.isoformat()} "
                            f"({missing} missing bars)"
                        )

                previous_open = open_time

    expected_rows = int(
        (
            END - START
        ).total_seconds()
        // INTERVAL.total_seconds()
    )

    expected_last_open = (
        END
        - INTERVAL
    )

    print()
    print("BTCUSDT 5m DATA AUDIT")
    print("---------------------")
    print(
        f"Monthly files:      "
        f"{len(files)} / "
        f"{len(expected_files)}"
    )
    print(
        f"Observed bars:      "
        f"{total_rows:,}"
    )
    print(
        f"Expected bars:      "
        f"{expected_rows:,}"
    )
    print(
        f"Missing bars:       "
        f"{gaps:,}"
    )
    print(
        f"Duplicate bars:     "
        f"{duplicates:,}"
    )
    print(
        f"Out of order:       "
        f"{out_of_order:,}"
    )
    print(
        f"Invalid rows:       "
        f"{invalid_rows:,}"
    )
    print(
        f"Irregular closes:   "
        f"{irregular_rows:,}"
    )
    print(
        f"First timestamp:    "
        f"{first_open}"
    )
    print(
        f"Last timestamp:     "
        f"{last_open}"
    )

    if missing_files:
        print(
            f"Missing files:      "
            f"{missing_files}"
        )

    if unexpected_files:
        print(
            f"Unexpected files:   "
            f"{unexpected_files}"
        )

    gap_count_reconciles = (
        expected_rows
        - total_rows
        == gaps
    )

    failures = (
        missing_files
        or unexpected_files
        or not gap_count_reconciles
        or duplicates != 0
        or out_of_order != 0
        or invalid_rows != 0
        or first_open != START
        or last_open != expected_last_open
    )

    if failures:
        raise SystemExit(
            "\nAUDIT FAILED"
        )

    print()
    print(
        "AUDIT PASSED — dataset structure is valid "
        "and source gaps are fully reconciled."
    )


if __name__ == "__main__":
    main()
