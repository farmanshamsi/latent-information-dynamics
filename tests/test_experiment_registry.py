from datetime import datetime, timezone

import json

import pytest

from latent_information_dynamics.experiments.registry import (
    ExperimentRegistry,
    ExperimentRegistryError,
    ExperimentResult,
)
from latent_information_dynamics.experiments.spec import (
    ExperimentSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    WalkForwardSpec,
)


UTC = timezone.utc


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        display_name="registry test",
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        dataset_fingerprint=
            "a" * 64,
        git_commit=
            "abcdef1234567890",
        experiment_group=
            "price_plus_information",
        features=(
            FeatureSpec.from_mapping(
                name="ofi",
                domain="microstructure",
                version="1.0.0",
            ),
        ),
        target=TargetSpec(
            name="btc_direction",
            horizon_seconds=900,
        ),
        model=ModelSpec.from_mapping(
            family="logistic_regression",
            version="1.0.0",
            parameters={
                "C": 1.0
            },
        ),
        validation=WalkForwardSpec(
            n_folds=4,
            min_train_observations=1000,
            test_observations=250,
            purge_observations=15,
            embargo_observations=15,
        ),
        random_seed=42,
    )


def test_register_creates_immutable_spec(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    item = spec()

    path = registry.register(
        item
    )

    assert path.is_file()

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["experiment_id"]
        == item.experiment_id
    )

    assert (
        payload["fingerprint"]
        == item.fingerprint
    )


def test_register_is_idempotent(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    item = spec()

    first = registry.register(
        item
    )

    second = registry.register(
        item
    )

    assert first == second


def test_list_ids(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    item = spec()

    registry.register(
        item
    )

    assert registry.list_ids() == (
        item.experiment_id,
    )


def test_result_requires_registered_experiment(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    result = ExperimentResult.from_mapping(
        experiment_id="exp-unknown",
        run_id="run-001",
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        metrics={
            "log_loss": 0.65
        },
    )

    with pytest.raises(
        ExperimentRegistryError
    ):
        registry.register_result(
            result
        )


def test_register_result(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    item = spec()

    registry.register(
        item
    )

    result = ExperimentResult.from_mapping(
        experiment_id=
            item.experiment_id,
        run_id="run-001",
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        metrics={
            "log_loss": 0.65,
            "brier": 0.23,
        },
        artifacts=(
            "predictions.parquet",
            "metrics.json",
        ),
    )

    path = registry.register_result(
        result
    )

    assert path.is_file()

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["metrics"]["log_loss"]
        == 0.65
    )

    assert (
        payload["experiment_id"]
        == item.experiment_id
    )


def test_result_run_id_is_immutable(
    tmp_path,
) -> None:
    registry = ExperimentRegistry(
        tmp_path
    )

    item = spec()

    registry.register(
        item
    )

    first = ExperimentResult.from_mapping(
        experiment_id=
            item.experiment_id,
        run_id="run-001",
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        metrics={
            "log_loss": 0.65
        },
    )

    second = ExperimentResult.from_mapping(
        experiment_id=
            item.experiment_id,
        run_id="run-001",
        created_at=datetime(
            2026,
            1,
            2,
            tzinfo=UTC,
        ),
        metrics={
            "log_loss": 0.55
        },
    )

    registry.register_result(
        first
    )

    with pytest.raises(
        ExperimentRegistryError
    ):
        registry.register_result(
            second
        )
