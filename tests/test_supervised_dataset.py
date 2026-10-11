from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.data.asof import (
    FeatureRequest,
)
from latent_information_dynamics.data.contracts import (
    NumericFeatureObservation,
)
from latent_information_dynamics.data.feature_frame import (
    build_feature_frame,
)
from latent_information_dynamics.data.timestamps import (
    InformationTimestamps,
)
from latent_information_dynamics.datasets.supervised import (
    SupervisedDatasetError,
    build_supervised_dataset,
    build_supervised_example,
)
from latent_information_dynamics.targets.binary import (
    DecisionWindow,
    ReferencePriceObservation,
    TieRule,
    build_binary_direction_target,
)


UTC = timezone.utc


def dt(
    day: int,
    hour: int,
    minute: int = 0,
) -> datetime:
    return datetime(
        2026,
        1,
        day,
        hour,
        minute,
        tzinfo=UTC,
    )


def feature_frame(
    *,
    day: int,
    hour: int,
    price_return: float,
    ofi: float,
):
    prediction_time = dt(
        day,
        hour,
    )

    observations = (
        NumericFeatureObservation(
            name="price_return_4h",
            value=price_return,
            source="research-feed",
            entity="BTC-USD",
            timestamps=InformationTimestamps(
                event_time=
                    prediction_time
                    - timedelta(
                        minutes=1
                    ),
                receive_time=
                    prediction_time,
                process_time=
                    prediction_time,
            ),
        ),
        NumericFeatureObservation(
            name="ofi_sum_1h",
            value=ofi,
            source="research-feed",
            entity="BTC-USD",
            timestamps=InformationTimestamps(
                event_time=
                    prediction_time
                    - timedelta(
                        seconds=1
                    ),
                receive_time=
                    prediction_time,
                process_time=
                    prediction_time,
            ),
        ),
    )

    return build_feature_frame(
        observations=observations,
        requests=(
            FeatureRequest(
                name="price_return_4h",
            ),
            FeatureRequest(
                name="ofi_sum_1h",
            ),
        ),
        prediction_time=
            prediction_time,
    )


def target(
    *,
    day: int,
    hour: int,
    start_price: float,
    end_price: float,
    end_available_delay_minutes: int = 1,
):
    decision_time = dt(
        day,
        hour,
    )

    resolution_time = (
        decision_time
        + timedelta(
            hours=4
        )
    )

    start = ReferencePriceObservation(
        source="btc-reference",
        reference_time=
            decision_time,
        available_time=
            decision_time
            + timedelta(
                seconds=2
            ),
        price=start_price,
    )

    end = ReferencePriceObservation(
        source="btc-reference",
        reference_time=
            resolution_time,
        available_time=
            resolution_time
            + timedelta(
                minutes=
                    end_available_delay_minutes
            ),
        price=end_price,
    )

    return build_binary_direction_target(
        window=DecisionWindow(
            decision_time=
                decision_time,
            resolution_time=
                resolution_time,
        ),
        start_reference=start,
        end_reference=end,
        tie_rule=TieRule.UP,
    )


def example(
    *,
    day: int,
    hour: int,
    price_return: float,
    ofi: float,
    start_price: float,
    end_price: float,
):
    return build_supervised_example(
        features=feature_frame(
            day=day,
            hour=hour,
            price_return=
                price_return,
            ofi=ofi,
        ),
        target=target(
            day=day,
            hour=hour,
            start_price=
                start_price,
            end_price=
                end_price,
        ),
    )


def test_supervised_example_contains_binary_target() -> None:
    item = example(
        day=1,
        hour=8,
        price_return=0.01,
        ofi=2.5,
        start_price=100.0,
        end_price=105.0,
    )

    assert item.target == 1

    assert item.feature_names == (
        "price_return_4h",
        "ofi_sum_1h",
    )


def test_feature_values_are_preserved() -> None:
    item = example(
        day=1,
        hour=8,
        price_return=0.01,
        ofi=2.5,
        start_price=100.0,
        end_price=105.0,
    )

    assert item.feature_dict() == {
        "price_return_4h": 0.01,
        "ofi_sum_1h": 2.5,
    }


def test_realized_return_is_preserved() -> None:
    item = example(
        day=1,
        hour=8,
        price_return=0.01,
        ofi=2.5,
        start_price=100.0,
        end_price=105.0,
    )

    assert (
        item.realized_return
        == pytest.approx(
            0.05
        )
    )


def test_label_availability_uses_future_reference_availability() -> None:
    item = build_supervised_example(
        features=feature_frame(
            day=1,
            hour=8,
            price_return=0.01,
            ofi=2.5,
        ),
        target=target(
            day=1,
            hour=8,
            start_price=100.0,
            end_price=105.0,
            end_available_delay_minutes=3,
        ),
    )

    assert (
        item.label_available_time
        == dt(
            1,
            12,
            3,
        )
    )


def test_feature_time_must_match_target_decision_time() -> None:
    with pytest.raises(
        SupervisedDatasetError
    ):
        build_supervised_example(
            features=feature_frame(
                day=1,
                hour=4,
                price_return=0.01,
                ofi=1.0,
            ),
            target=target(
                day=1,
                hour=8,
                start_price=100.0,
                end_price=105.0,
            ),
        )


def test_excluded_target_cannot_enter_dataset() -> None:
    decision_time = dt(
        1,
        8,
    )

    resolution_time = dt(
        1,
        12,
    )

    excluded = build_binary_direction_target(
        window=DecisionWindow(
            decision_time=
                decision_time,
            resolution_time=
                resolution_time,
        ),
        start_reference=
            ReferencePriceObservation(
                source="btc-reference",
                reference_time=
                    decision_time,
                available_time=
                    decision_time,
                price=100.0,
            ),
        end_reference=
            ReferencePriceObservation(
                source="btc-reference",
                reference_time=
                    resolution_time,
                available_time=
                    resolution_time,
                price=100.0,
            ),
        tie_rule=TieRule.EXCLUDE,
    )

    with pytest.raises(
        SupervisedDatasetError
    ):
        build_supervised_example(
            features=feature_frame(
                day=1,
                hour=8,
                price_return=0.0,
                ofi=0.0,
            ),
            target=excluded,
        )


def test_dataset_sorts_examples_chronologically() -> None:
    later = example(
        day=1,
        hour=12,
        price_return=0.02,
        ofi=3.0,
        start_price=105.0,
        end_price=106.0,
    )

    earlier = example(
        day=1,
        hour=8,
        price_return=0.01,
        ofi=2.0,
        start_price=100.0,
        end_price=105.0,
    )

    dataset = build_supervised_dataset(
        (
            (
                feature_frame(
                    day=1,
                    hour=12,
                    price_return=0.02,
                    ofi=3.0,
                ),
                target(
                    day=1,
                    hour=12,
                    start_price=105.0,
                    end_price=106.0,
                ),
            ),
            (
                feature_frame(
                    day=1,
                    hour=8,
                    price_return=0.01,
                    ofi=2.0,
                ),
                target(
                    day=1,
                    hour=8,
                    start_price=100.0,
                    end_price=105.0,
                ),
            ),
        )
    )

    assert (
        dataset.examples[0].decision_time
        == earlier.decision_time
    )

    assert (
        dataset.examples[1].decision_time
        == later.decision_time
    )


def test_duplicate_decision_times_are_rejected() -> None:
    pair = (
        feature_frame(
            day=1,
            hour=8,
            price_return=0.01,
            ofi=2.0,
        ),
        target(
            day=1,
            hour=8,
            start_price=100.0,
            end_price=105.0,
        ),
    )

    with pytest.raises(
        SupervisedDatasetError
    ):
        build_supervised_dataset(
            (
                pair,
                pair,
            )
        )


def test_matrix_preserves_requested_feature_order() -> None:
    dataset = build_supervised_dataset(
        (
            (
                feature_frame(
                    day=1,
                    hour=8,
                    price_return=0.01,
                    ofi=2.0,
                ),
                target(
                    day=1,
                    hour=8,
                    start_price=100.0,
                    end_price=105.0,
                ),
            ),
        )
    )

    matrix = dataset.matrix(
        features=(
            "ofi_sum_1h",
            "price_return_4h",
        )
    )

    assert matrix.feature_names == (
        "ofi_sum_1h",
        "price_return_4h",
    )

    assert matrix.X == (
        (
            2.0,
            0.01,
        ),
    )

    assert matrix.y == (
        1,
    )


def test_unknown_matrix_feature_is_rejected() -> None:
    dataset = build_supervised_dataset(
        (
            (
                feature_frame(
                    day=1,
                    hour=8,
                    price_return=0.01,
                    ofi=2.0,
                ),
                target(
                    day=1,
                    hour=8,
                    start_price=100.0,
                    end_price=105.0,
                ),
            ),
        )
    )

    with pytest.raises(
        SupervisedDatasetError
    ):
        dataset.matrix(
            features=(
                "does_not_exist",
            )
        )


def test_training_cutoff_excludes_unresolved_future_label() -> None:
    dataset = build_supervised_dataset(
        (
            (
                feature_frame(
                    day=1,
                    hour=8,
                    price_return=0.01,
                    ofi=2.0,
                ),
                target(
                    day=1,
                    hour=8,
                    start_price=100.0,
                    end_price=105.0,
                ),
            ),
            (
                feature_frame(
                    day=1,
                    hour=12,
                    price_return=-0.01,
                    ofi=-2.0,
                ),
                target(
                    day=1,
                    hour=12,
                    start_price=105.0,
                    end_price=100.0,
                ),
            ),
        )
    )

    available = (
        dataset.available_for_training_by(
            dt(
                1,
                15,
            )
        )
    )

    assert available is not None

    assert len(
        available.examples
    ) == 1

    assert (
        available.examples[0].decision_time
        == dt(
            1,
            8,
        )
    )


def test_no_labels_available_returns_none() -> None:
    dataset = build_supervised_dataset(
        (
            (
                feature_frame(
                    day=1,
                    hour=8,
                    price_return=0.01,
                    ofi=2.0,
                ),
                target(
                    day=1,
                    hour=8,
                    start_price=100.0,
                    end_price=105.0,
                ),
            ),
        )
    )

    assert (
        dataset.available_for_training_by(
            dt(
                1,
                10,
            )
        )
        is None
    )
