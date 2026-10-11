from datetime import datetime, timedelta, timezone

import pytest

from latent_information_dynamics.features.microstructure.price_independent import (
    DEFAULT_WINDOW_LABELS,
    PriceIndependentFeatureError,
    extract_strict_price_independent_features,
    strict_feature_names,
)
from latent_information_dynamics.features.microstructure.rolling import (
    DEFAULT_WINDOWS,
    MicrostructureWindowSummary,
)


UTC = timezone.utc


def make_summary(
    label: str,
    *,
    available_time: datetime,
    offset: float = 0.0,
) -> MicrostructureWindowSummary:
    return MicrostructureWindowSummary(
        label=label,
        window=DEFAULT_WINDOWS[label],
        available_time=available_time,
        observation_count=100,

        cumulative_midpoint_log_return=9.0,
        realized_volatility=8.0,

        ofi_sum=7.0,
        ofi_mean=6.0,

        relative_spread_bps_mean=5.0,
        relative_spread_bps_max=5.5,

        depth_imbalance_mean=
            0.10 + offset,
        depth_imbalance_last=
            0.20 + offset,

        top_level_imbalance_mean=
            0.30 + offset,
        top_level_imbalance_last=
            0.40 + offset,

        microprice_displacement_bps_mean=4.0,
        microprice_displacement_bps_last=4.5,

        trade_imbalance_mean=
            0.50 + offset,
        trade_imbalance_last=
            0.60 + offset,

        notional_imbalance_mean=3.0,
        notional_imbalance_last=3.5,
    )


def test_strict_contract_has_36_features() -> None:
    names = strict_feature_names()

    assert len(names) == 36
    assert len(set(names)) == 36

    assert (
        names[0]
        == "depth_imbalance_mean_1m"
    )

    assert (
        names[-1]
        == "trade_imbalance_last_4h"
    )


def test_strict_contract_excludes_price_derived_features() -> None:
    names = strict_feature_names()

    forbidden = (
        "midpoint",
        "volatility",
        "spread",
        "microprice",
        "ofi",
        "notional",
    )

    for name in names:
        assert not any(
            token in name
            for token in forbidden
        )


def test_extracts_only_quantity_and_trade_size_information() -> None:
    timestamp = datetime(
        2025, 1, 1, 12, 0,
        tzinfo=UTC,
    )

    summaries = {
        label: make_summary(
            label,
            available_time=timestamp,
            offset=index * 0.01,
        )
        for index, label
        in enumerate(
            DEFAULT_WINDOW_LABELS
        )
    }

    vector = (
        extract_strict_price_independent_features(
            summaries
        )
    )

    assert vector.available_time == timestamp
    assert len(vector.feature_names) == 36
    assert len(vector.feature_values) == 36

    values = vector.as_dict()

    assert (
        values[
            "depth_imbalance_mean_1m"
        ]
        == 0.10
    )

    assert (
        values[
            "trade_imbalance_last_4h"
        ]
        == 0.65
    )


def test_rejects_different_availability_times() -> None:
    timestamp = datetime(
        2025, 1, 1, 12, 0,
        tzinfo=UTC,
    )

    summaries = {
        label: make_summary(
            label,
            available_time=timestamp,
        )
        for label in DEFAULT_WINDOW_LABELS
    }

    summaries["4h"] = make_summary(
        "4h",
        available_time=
            timestamp
            + timedelta(seconds=1),
    )

    with pytest.raises(
        PriceIndependentFeatureError
    ):
        extract_strict_price_independent_features(
            summaries
        )
