from datetime import datetime, timedelta, timezone

import math

import pytest

from latent_information_dynamics.features.microstructure.engine import (
    MicrostructureState,
)
from latent_information_dynamics.features.microstructure.rolling import (
    MultiHorizonMicrostructureEngine,
    RollingFeatureError,
    summaries_as_feature_observations,
)


UTC = timezone.utc


def ts(
    minute: int,
) -> datetime:
    return datetime(
        2026,
        1,
        1,
        12,
        minute,
        tzinfo=UTC,
    )


def state(
    *,
    minute: int,
    return_value: float = 0.0,
    ofi: float = 0.0,
    spread_bps: float = 10.0,
    depth_imbalance: float = 0.0,
    top_imbalance: float = 0.0,
    microprice_bps: float = 0.0,
    trade_imbalance: float = 0.0,
    notional_imbalance: float = 0.0,
) -> MicrostructureState:
    return MicrostructureState(
        event_time=ts(minute),
        available_time=ts(minute),

        midpoint=100.0,
        midpoint_log_return=
            return_value,

        spread=0.1,
        relative_spread_bps=
            spread_bps,

        top_level_imbalance=
            top_imbalance,

        depth_imbalance=
            depth_imbalance,

        microprice=100.0,
        microprice_displacement_bps=
            microprice_bps,

        order_flow_imbalance=
            ofi,

        trade_count=1,

        traded_size=1.0,
        signed_traded_size=
            trade_imbalance,

        trade_imbalance=
            trade_imbalance,

        traded_notional=100.0,
        signed_traded_notional=
            100.0 * notional_imbalance,

        notional_imbalance=
            notional_imbalance,
    )


def test_single_state_summary() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    summaries = engine.update(
        state(
            minute=0,
            ofi=2.0,
        )
    )

    summary = summaries["5m"]

    assert summary.observation_count == 1
    assert summary.ofi_sum == 2.0
    assert summary.ofi_mean == 2.0


def test_ofi_is_accumulated() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            ofi=1.0,
        )
    )

    result = engine.update(
        state(
            minute=1,
            ofi=3.0,
        )
    )

    summary = result["5m"]

    assert summary.ofi_sum == 4.0
    assert summary.ofi_mean == 2.0


def test_log_returns_are_accumulated() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            return_value=0.01,
        )
    )

    result = engine.update(
        state(
            minute=1,
            return_value=-0.002,
        )
    )

    assert (
        result["5m"].
        cumulative_midpoint_log_return
        == pytest.approx(
            0.008
        )
    )


def test_realized_volatility_uses_squared_returns() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            return_value=0.03,
        )
    )

    result = engine.update(
        state(
            minute=1,
            return_value=0.04,
        )
    )

    assert (
        result["5m"].
        realized_volatility
        == pytest.approx(
            0.05
        )
    )


def test_spread_uses_mean_and_max() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            spread_bps=10.0,
        )
    )

    result = engine.update(
        state(
            minute=1,
            spread_bps=20.0,
        )
    )

    summary = result["5m"]

    assert (
        summary.relative_spread_bps_mean
        == 15.0
    )

    assert (
        summary.relative_spread_bps_max
        == 20.0
    )


def test_state_features_have_mean_and_latest() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            depth_imbalance=-0.5,
        )
    )

    result = engine.update(
        state(
            minute=1,
            depth_imbalance=0.5,
        )
    )

    summary = result["5m"]

    assert (
        summary.depth_imbalance_mean
        == 0.0
    )

    assert (
        summary.depth_imbalance_last
        == 0.5
    )


def test_short_window_excludes_old_states() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
        }
    )

    engine.update(
        state(
            minute=0,
            ofi=100.0,
        )
    )

    result = engine.update(
        state(
            minute=3,
            ofi=1.0,
        )
    )

    assert (
        result["1m"].ofi_sum
        == 1.0
    )

    assert (
        result["5m"].ofi_sum
        == 101.0
    )


def test_boundary_timestamp_is_included() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            ofi=2.0,
        )
    )

    result = engine.update(
        state(
            minute=5,
            ofi=3.0,
        )
    )

    assert (
        result["5m"].ofi_sum
        == 5.0
    )


def test_state_older_than_max_window_is_pruned() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=0,
            ofi=100.0,
        )
    )

    result = engine.update(
        state(
            minute=6,
            ofi=1.0,
        )
    )

    assert (
        result["5m"].ofi_sum
        == 1.0
    )


def test_time_cannot_move_backwards() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    engine.update(
        state(
            minute=2
        )
    )

    with pytest.raises(
        RollingFeatureError
    ):
        engine.update(
            state(
                minute=1
            )
        )


def test_multiple_horizons_produce_distinct_states() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
        }
    )

    engine.update(
        state(
            minute=0,
            ofi=10.0,
        )
    )

    result = engine.update(
        state(
            minute=3,
            ofi=2.0,
        )
    )

    assert result["1m"].ofi_sum == 2.0
    assert result["5m"].ofi_sum == 12.0


def test_feature_observation_names_include_horizon() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    summaries = engine.update(
        state(
            minute=0
        )
    )

    observations = summaries_as_feature_observations(
        summaries,
        source="binance-live",
        entity="BTC-USDT",
    )

    names = {
        item.name
        for item in observations
    }

    assert "ofi_sum_5m" in names

    assert (
        "realized_volatility_5m"
        in names
    )

    assert (
        "microprice_displacement_bps_last_5m"
        in names
    )


def test_feature_observations_are_point_in_time_safe() -> None:
    engine = MultiHorizonMicrostructureEngine(
        windows={
            "5m": timedelta(minutes=5)
        }
    )

    summaries = engine.update(
        state(
            minute=4
        )
    )

    observations = summaries_as_feature_observations(
        summaries,
        source="binance-live",
        entity="BTC-USDT",
    )

    for observation in observations:
        assert observation.available_time == ts(4)
