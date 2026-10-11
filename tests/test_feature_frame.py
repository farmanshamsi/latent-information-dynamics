from datetime import datetime, timezone

import pytest

from latent_information_dynamics.data.asof import (
    FeatureRequest,
)
from latent_information_dynamics.data.contracts import (
    NumericFeatureObservation,
)
from latent_information_dynamics.data.feature_frame import (
    FeatureFrameError,
    build_feature_frame,
    derive_numeric_feature,
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


def obs(
    *,
    name: str,
    value: float,
    event_minute: int,
    available_minute: int,
    source: str = "exchange",
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


def base_row():
    observations = [
        obs(
            name="bid",
            value=100.0,
            event_minute=1,
            available_minute=2,
        ),
        obs(
            name="ask",
            value=102.0,
            event_minute=2,
            available_minute=4,
        ),
    ]

    requests = [
        FeatureRequest(
            name="bid",
        ),
        FeatureRequest(
            name="ask",
        ),
    ]

    return build_feature_frame(
        observations,
        requests,
        prediction_time=ts(5),
    )


def test_raw_feature_values_are_preserved() -> None:
    row = base_row()

    assert row.values() == {
        "bid": 100.0,
        "ask": 102.0,
    }


def test_raw_feature_has_source_provenance() -> None:
    row = base_row()

    provenance = row.get(
        "bid"
    ).provenance

    assert len(
        provenance.raw_inputs
    ) == 1

    raw = provenance.raw_inputs[0]

    assert raw.source == "exchange"
    assert raw.entity == "BTC-USD"
    assert raw.event_time == ts(1)
    assert raw.available_time == ts(2)


def test_asof_transformation_is_recorded() -> None:
    row = base_row()

    transformation = (
        row.get(
            "bid"
        )
        .provenance
        .transformation
    )

    assert transformation.name == "asof_select"
    assert transformation.version == "1.0.0"


def test_derived_midprice() -> None:
    row = base_row()

    row = derive_numeric_feature(
        row,
        output_name="midprice",
        input_names=(
            "bid",
            "ask",
        ),
        function=lambda bid, ask:
            (bid + ask) / 2.0,
        transformation_name="midprice",
        transformation_version="1.0.0",
    )

    assert row.values()[
        "midprice"
    ] == 101.0


def test_derived_feature_availability_is_latest_input() -> None:
    row = base_row()

    row = derive_numeric_feature(
        row,
        output_name="spread",
        input_names=(
            "bid",
            "ask",
        ),
        function=lambda bid, ask:
            ask - bid,
        transformation_name="spread",
        transformation_version="1.0.0",
    )

    feature = row.get(
        "spread"
    )

    assert feature.available_time == ts(4)


def test_derived_feature_retains_both_raw_inputs() -> None:
    row = base_row()

    row = derive_numeric_feature(
        row,
        output_name="spread",
        input_names=(
            "bid",
            "ask",
        ),
        function=lambda bid, ask:
            ask - bid,
        transformation_name="spread",
        transformation_version="1.0.0",
    )

    raw_names = {
        item.observation_name
        for item in row.get(
            "spread"
        ).provenance.raw_inputs
    }

    assert raw_names == {
        "bid",
        "ask",
    }


def test_chained_feature_retains_original_raw_lineage() -> None:
    row = base_row()

    row = derive_numeric_feature(
        row,
        output_name="midprice",
        input_names=(
            "bid",
            "ask",
        ),
        function=lambda bid, ask:
            (bid + ask) / 2.0,
        transformation_name="midprice",
        transformation_version="1.0.0",
    )

    row = derive_numeric_feature(
        row,
        output_name="scaled_mid",
        input_names=(
            "midprice",
        ),
        function=lambda value:
            value / 100.0,
        transformation_name="scale",
        transformation_version="1.0.0",
        parameters={
            "divisor": 100.0
        },
    )

    raw_names = {
        item.observation_name
        for item in row.get(
            "scaled_mid"
        ).provenance.raw_inputs
    }

    assert raw_names == {
        "bid",
        "ask",
    }


def test_provenance_fingerprint_is_deterministic() -> None:
    row_1 = base_row()
    row_2 = base_row()

    fingerprint_1 = (
        row_1.get(
            "bid"
        )
        .provenance
        .fingerprint
    )

    fingerprint_2 = (
        row_2.get(
            "bid"
        )
        .provenance
        .fingerprint
    )

    assert fingerprint_1 == fingerprint_2


def test_transformation_parameters_change_fingerprint() -> None:
    row = base_row()

    first = derive_numeric_feature(
        row,
        output_name="scaled_a",
        input_names=("bid",),
        function=lambda x: x / 10,
        transformation_name="scale",
        transformation_version="1.0.0",
        parameters={
            "divisor": 10
        },
    )

    second = derive_numeric_feature(
        row,
        output_name="scaled_b",
        input_names=("bid",),
        function=lambda x: x / 20,
        transformation_name="scale",
        transformation_version="1.0.0",
        parameters={
            "divisor": 20
        },
    )

    fingerprint_a = (
        first.get(
            "scaled_a"
        )
        .provenance
        .fingerprint
    )

    fingerprint_b = (
        second.get(
            "scaled_b"
        )
        .provenance
        .fingerprint
    )

    assert fingerprint_a != fingerprint_b


def test_missing_input_feature_raises() -> None:
    row = base_row()

    with pytest.raises(KeyError):
        derive_numeric_feature(
            row,
            output_name="bad",
            input_names=(
                "does_not_exist",
            ),
            function=lambda x: x,
            transformation_name="identity",
            transformation_version="1.0.0",
        )


def test_duplicate_feature_name_is_rejected() -> None:
    row = base_row()

    with pytest.raises(
        FeatureFrameError
    ):
        derive_numeric_feature(
            row,
            output_name="bid",
            input_names=("ask",),
            function=lambda x: x,
            transformation_name="duplicate",
            transformation_version="1.0.0",
        )


def test_nonfinite_derived_value_is_rejected() -> None:
    row = base_row()

    with pytest.raises(
        FeatureFrameError
    ):
        derive_numeric_feature(
            row,
            output_name="bad",
            input_names=("bid",),
            function=lambda x:
                float("nan"),
            transformation_name="bad-transform",
            transformation_version="1.0.0",
        )
