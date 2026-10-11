from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.data.timestamps import (
    InformationTimestamps,
    TimestampError,
)


UTC = timezone.utc


def ts(minute: int) -> datetime:
    return datetime(
        2026,
        1,
        1,
        12,
        minute,
        tzinfo=UTC,
    )


def test_available_time_uses_latest_constraint() -> None:
    timestamps = InformationTimestamps(
        event_time=ts(0),
        publish_time=ts(1),
        receive_time=ts(2),
        process_time=ts(3),
    )

    assert timestamps.available_time == ts(3)


def test_source_specific_availability_can_delay_record() -> None:
    timestamps = InformationTimestamps(
        event_time=ts(0),
        publish_time=ts(1),
        receive_time=ts(2),
        process_time=ts(3),
        source_available_time=ts(5),
    )

    assert timestamps.available_time == ts(5)


def test_event_time_does_not_define_availability() -> None:
    timestamps = InformationTimestamps(
        event_time=ts(0),
        receive_time=ts(4),
        process_time=ts(5),
    )

    assert timestamps.available_time == ts(5)
    assert not timestamps.is_available_by(ts(4))
    assert timestamps.is_available_by(ts(5))


def test_naive_datetime_is_rejected() -> None:
    naive = datetime(2026, 1, 1, 12, 0)

    with pytest.raises(TimestampError):
        InformationTimestamps(
            event_time=naive,
            receive_time=ts(1),
            process_time=ts(2),
        )


def test_non_utc_aware_timestamp_is_normalized() -> None:
    plus_five = timezone(timedelta(hours=5))

    local_event = datetime(
        2026,
        1,
        1,
        17,
        0,
        tzinfo=plus_five,
    )

    timestamps = InformationTimestamps(
        event_time=local_event,
        receive_time=ts(1),
        process_time=ts(2),
    )

    assert timestamps.event_time == ts(0)


def test_latency_seconds() -> None:
    timestamps = InformationTimestamps(
        event_time=ts(0),
        receive_time=ts(1),
        process_time=ts(2),
    )

    assert timestamps.latency_seconds() == 120.0


def test_audit_detects_suspicious_ordering() -> None:
    timestamps = InformationTimestamps(
        event_time=ts(2),
        publish_time=ts(1),
        receive_time=ts(0),
        process_time=ts(3),
    )

    violations = timestamps.audit()

    assert "publish_time precedes event_time" in violations
    assert "receive_time precedes publish_time" in violations
