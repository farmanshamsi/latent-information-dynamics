from datetime import datetime, timezone

import pytest

from latent_information_dynamics.data.contracts import (
    FeatureObservation,
    NumericFeatureObservation,
    PointInTimeViolation,
    latest_available,
)
from latent_information_dynamics.data.timestamps import (
    InformationTimestamps,
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


def observation(
    *,
    value: float,
    available_minute: int,
) -> NumericFeatureObservation:
    return NumericFeatureObservation(
        name="order_flow_imbalance",
        value=value,
        source="exchange-feed",
        entity="BTC-USD",
        timestamps=InformationTimestamps(
            event_time=ts(available_minute - 2),
            receive_time=ts(available_minute - 1),
            process_time=ts(available_minute),
        ),
    )


def test_feature_available_after_processing() -> None:
    obs = observation(
        value=0.42,
        available_minute=5,
    )

    assert not obs.is_available_by(ts(4))
    assert obs.is_available_by(ts(5))


def test_future_feature_raises_point_in_time_violation() -> None:
    obs = observation(
        value=0.42,
        available_minute=5,
    )

    with pytest.raises(PointInTimeViolation):
        obs.require_available_by(ts(4))


def test_legal_feature_passes_point_in_time_check() -> None:
    obs = observation(
        value=0.42,
        available_minute=5,
    )

    assert obs.require_available_by(ts(5)) is obs


def test_latest_available_excludes_future_information() -> None:
    observations = [
        observation(
            value=1.0,
            available_minute=2,
        ),
        observation(
            value=2.0,
            available_minute=4,
        ),
        observation(
            value=999.0,
            available_minute=8,
        ),
    ]

    result = latest_available(
        observations,
        prediction_time=ts(5),
    )

    assert result is not None
    assert result.value == 2.0


def test_latest_available_returns_none_when_nothing_is_legal() -> None:
    observations = [
        observation(
            value=1.0,
            available_minute=6,
        )
    ]

    assert latest_available(
        observations,
        prediction_time=ts(5),
    ) is None


def test_numeric_feature_rejects_nan() -> None:
    with pytest.raises(ValueError):
        NumericFeatureObservation(
            name="bad_feature",
            value=float("nan"),
            source="test",
            timestamps=InformationTimestamps(
                event_time=ts(0),
                receive_time=ts(1),
                process_time=ts(2),
            ),
        )


def test_empty_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureObservation(
            name="feature",
            value="value",
            source="",
            timestamps=InformationTimestamps(
                event_time=ts(0),
                receive_time=ts(1),
                process_time=ts(2),
            ),
        )
