from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.data.asof import (
    AmbiguousObservationError,
    FeatureRequest,
    MissingRequiredFeature,
    asof_join,
    select_asof,
)
from latent_information_dynamics.data.contracts import (
    NumericFeatureObservation,
)
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


def observation(
    *,
    name: str = "feature",
    value: float,
    event_minute: int,
    available_minute: int,
    source: str = "source-a",
    entity: str = "BTC-USD",
) -> NumericFeatureObservation:
    return NumericFeatureObservation(
        name=name,
        value=value,
        source=source,
        entity=entity,
        timestamps=InformationTimestamps(
            event_time=ts(event_minute),
            receive_time=ts(
                available_minute - 1
            ),
            process_time=ts(
                available_minute
            ),
        ),
    )


def test_selects_latest_legally_available_observation() -> None:
    observations = [
        observation(
            value=1.0,
            event_minute=0,
            available_minute=2,
        ),
        observation(
            value=2.0,
            event_minute=2,
            available_minute=4,
        ),
        observation(
            value=999.0,
            event_minute=4,
            available_minute=8,
        ),
    ]

    selected = select_asof(
        observations,
        FeatureRequest(
            name="feature",
        ),
        prediction_time=ts(5),
    )

    assert selected is not None
    assert selected.value == 2.0
    assert selected.available_time == ts(4)


def test_future_information_is_excluded() -> None:
    observations = [
        observation(
            value=1.0,
            event_minute=0,
            available_minute=2,
        ),
        observation(
            value=999.0,
            event_minute=1,
            available_minute=6,
        ),
    ]

    selected = select_asof(
        observations,
        FeatureRequest(
            name="feature",
        ),
        prediction_time=ts(5),
    )

    assert selected is not None
    assert selected.value == 1.0


def test_historical_revision_is_point_in_time_safe() -> None:
    observations = [
        observation(
            name="macro_release",
            value=100.0,
            event_minute=0,
            available_minute=2,
            source="macro-provider",
            entity="indicator-x",
        ),
        observation(
            name="macro_release",
            value=101.5,
            event_minute=0,
            available_minute=8,
            source="macro-provider",
            entity="indicator-x",
        ),
    ]

    request = FeatureRequest(
        name="macro_release",
        source="macro-provider",
        entity="indicator-x",
    )

    early = select_asof(
        observations,
        request,
        prediction_time=ts(5),
    )

    late = select_asof(
        observations,
        request,
        prediction_time=ts(9),
    )

    assert early is not None
    assert late is not None

    assert early.value == 100.0
    assert late.value == 101.5


def test_source_filter_is_respected() -> None:
    observations = [
        observation(
            value=1.0,
            event_minute=0,
            available_minute=2,
            source="source-a",
        ),
        observation(
            value=2.0,
            event_minute=0,
            available_minute=3,
            source="source-b",
        ),
    ]

    selected = select_asof(
        observations,
        FeatureRequest(
            name="feature",
            source="source-a",
        ),
        prediction_time=ts(5),
    )

    assert selected is not None
    assert selected.source == "source-a"
    assert selected.value == 1.0


def test_entity_filter_is_respected() -> None:
    observations = [
        observation(
            value=10.0,
            event_minute=0,
            available_minute=2,
            entity="BTC-USD",
        ),
        observation(
            value=20.0,
            event_minute=0,
            available_minute=3,
            entity="ETH-USD",
        ),
    ]

    selected = select_asof(
        observations,
        FeatureRequest(
            name="feature",
            entity="BTC-USD",
        ),
        prediction_time=ts(5),
    )

    assert selected is not None
    assert selected.entity == "BTC-USD"
    assert selected.value == 10.0


def test_stale_required_feature_is_rejected() -> None:
    observations = [
        observation(
            value=1.0,
            event_minute=0,
            available_minute=1,
        )
    ]

    request = FeatureRequest(
        name="feature",
        max_age=timedelta(minutes=2),
    )

    with pytest.raises(
        MissingRequiredFeature
    ):
        asof_join(
            observations,
            [request],
            prediction_time=ts(5),
        )


def test_optional_missing_feature_is_recorded() -> None:
    result = asof_join(
        observations=[],
        requests=[
            FeatureRequest(
                name="news_signal",
                required=False,
            )
        ],
        prediction_time=ts(5),
    )

    assert result.values() == {}
    assert result.missing_optional == (
        "news_signal",
    )


def test_output_alias_is_used() -> None:
    observations = [
        observation(
            name="order_flow_imbalance",
            value=0.42,
            event_minute=2,
            available_minute=3,
        )
    ]

    result = asof_join(
        observations,
        [
            FeatureRequest(
                name="order_flow_imbalance",
                output_name="ofi",
            )
        ],
        prediction_time=ts(5),
    )

    assert result.values() == {
        "ofi": 0.42
    }


def test_event_age_and_availability_age_are_distinct() -> None:
    observations = [
        observation(
            value=5.0,
            event_minute=0,
            available_minute=8,
        )
    ]

    request_available = FeatureRequest(
        name="feature",
        max_age=timedelta(minutes=3),
        age_basis="available",
    )

    selected = select_asof(
        observations,
        request_available,
        prediction_time=ts(10),
    )

    assert selected is not None

    request_event = FeatureRequest(
        name="feature",
        max_age=timedelta(minutes=3),
        age_basis="event",
    )

    rejected = select_asof(
        observations,
        request_event,
        prediction_time=ts(10),
    )

    assert rejected is None


def test_equal_latest_timestamps_raise_ambiguity() -> None:
    observations = [
        observation(
            value=1.0,
            event_minute=1,
            available_minute=3,
        ),
        observation(
            value=2.0,
            event_minute=2,
            available_minute=3,
        ),
    ]

    with pytest.raises(
        AmbiguousObservationError
    ):
        select_asof(
            observations,
            FeatureRequest(
                name="feature",
            ),
            prediction_time=ts(5),
        )


def test_naive_prediction_time_is_rejected() -> None:
    naive = datetime(
        2026,
        1,
        1,
        12,
        5,
    )

    with pytest.raises(TimestampError):
        asof_join(
            observations=[],
            requests=[],
            prediction_time=naive,
        )
