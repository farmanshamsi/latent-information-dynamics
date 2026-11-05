from __future__ import annotations

import csv

from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc

ROOT = Path(
    "data/raw/binance/BTCUSDT/5m"
)

START = datetime(
    2026, 1, 1,
    tzinfo=UTC,
)

END = datetime(
    2026, 7, 1,
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

    cursor = START

    while cursor < END:
        names.append(
            f"{cursor:%Y-%m}.csv"
        )

        if cursor.month == 12:
            cursor = datetime(
                cursor.year + 1,
                1,
                1,
                tzinfo=UTC,
            )
        else:
            cursor = datetime(
                cursor.year,
                cursor.month + 1,
                1,
                tzinfo=UTC,
            )

    return names


def main() -> None:
    expected_files = (
        expected_month_names()
    )

    paths = [
        ROOT / name
        for name in expected_files
    ]

    missing_files = [
        path.name
        for path in paths
        if not path.exists()
    ]

    if missing_files:
        print(
            "Missing holdout files:",
            missing_files,
        )

        raise SystemExit(
            "\nHOLDOUT AUDIT FAILED"
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

    for path in paths:
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
                            row[
                                "open_time"
                            ]
                        )
                    )

                    close_time = (
                        datetime.fromisoformat(
                            row[
                                "close_time"
                            ]
                        )
                    )

                except Exception:
                    invalid_rows += 1

                    print(
                        "BAD TIMESTAMP "
                        f"{path}:{row_number}"
                    )

                    continue

                if first_open is None:
                    first_open = (
                        open_time
                    )

                last_open = (
                    open_time
                )

                valid = True

                if (
                    row[
                        "symbol"
                    ]
                    != "BTCUSDT"
                ):
                    valid = False

                if (
                    row[
                        "interval"
                    ]
                    != "5m"
                ):
                    valid = False

                if (
                    open_time.tzinfo
                    is None
                ):
                    valid = False

                nominal_close = (
                    open_time
                    + BAR_CLOSE_OFFSET
                )

                slot_end = (
                    open_time
                    + INTERVAL
                )

                if (
                    close_time
                    != nominal_close
                ):
                    if (
                        open_time
                        < close_time
                        < slot_end
                    ):
                        irregular_rows += 1

                        print(
                            "IRREGULAR CLOSE "
                            f"{path}:{row_number} "
                            f"{open_time.isoformat()} "
                            "→ "
                            f"{close_time.isoformat()}"
                        )

                    else:
                        valid = False

                if (
                    open_time.second != 0
                    or open_time.microsecond
                    != 0
                    or open_time.minute % 5
                    != 0
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
                        "INVALID ROW "
                        f"{path}:{row_number} "
                        f"{open_time.isoformat()}"
                    )

                if (
                    previous_open
                    is not None
                ):
                    expected = (
                        previous_open
                        + INTERVAL
                    )

                    if (
                        open_time
                        == previous_open
                    ):
                        duplicates += 1

                    elif (
                        open_time
                        < expected
                    ):
                        out_of_order += 1

                    elif (
                        open_time
                        > expected
                    ):
                        missing = int(
                            (
                                open_time
                                - expected
                            ).total_seconds()
                            // INTERVAL.total_seconds()
                        )

                        gaps += (
                            missing
                        )

                        print(
                            "GAP: expected "
                            f"{expected.isoformat()}, "
                            "received "
                            f"{open_time.isoformat()} "
                            f"({missing} missing bars)"
                        )

                previous_open = (
                    open_time
                )

    expected_rows = int(
        (
            END
            - START
        ).total_seconds()
        // INTERVAL.total_seconds()
    )

    expected_last_open = (
        END
        - INTERVAL
    )

    print()
    print(
        "BTCUSDT 5m 2026 HOLDOUT AUDIT"
    )
    print(
        "-----------------------------"
    )

    print(
        "Monthly files:      "
        f"{len(paths)} / "
        f"{len(expected_files)}"
    )

    print(
        "Observed bars:      "
        f"{total_rows:,}"
    )

    print(
        "Expected bars:      "
        f"{expected_rows:,}"
    )

    print(
        "Missing bars:       "
        f"{gaps:,}"
    )

    print(
        "Duplicate bars:     "
        f"{duplicates:,}"
    )

    print(
        "Out of order:       "
        f"{out_of_order:,}"
    )

    print(
        "Invalid rows:       "
        f"{invalid_rows:,}"
    )

    print(
        "Irregular closes:   "
        f"{irregular_rows:,}"
    )

    print(
        "First timestamp:    "
        f"{first_open}"
    )

    print(
        "Last timestamp:     "
        f"{last_open}"
    )

    gap_count_reconciles = (
        expected_rows
        - total_rows
        == gaps
    )

    failures = (
        not gap_count_reconciles
        or duplicates != 0
        or out_of_order != 0
        or invalid_rows != 0
        or first_open != START
        or last_open
        != expected_last_open
    )

    if failures:
        raise SystemExit(
            "\nHOLDOUT AUDIT FAILED"
        )

    print()
    print(
        "HOLDOUT AUDIT PASSED — "
        "2026 source structure is valid "
        "and source gaps are fully reconciled."
    )


if __name__ == "__main__":
    main()
