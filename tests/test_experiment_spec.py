from datetime import datetime, timezone

import pytest

from latent_information_dynamics.experiments.spec import (
    ExperimentSpec,
    ExperimentSpecError,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    WalkForwardSpec,
)


UTC = timezone.utc


def feature(
    *,
    name: str = "ofi",
    parameter: float = 1.0,
) -> FeatureSpec:
    return FeatureSpec.from_mapping(
        name=name,
        domain="microstructure",
        version="1.0.0",
        parameters={
            "scale": parameter
        },
    )


def experiment(
    *,
    created_at: datetime,
    dataset_fingerprint: str = "a" * 64,
    feature_parameter: float = 1.0,
    feature_order: tuple[str, ...] = (
        "ofi",
        "spread",
    ),
) -> ExperimentSpec:
    features = tuple(
        feature(
            name=name,
            parameter=feature_parameter,
        )
        for name in feature_order
    )

    return ExperimentSpec(
        display_name="test experiment",
        created_at=created_at,
        dataset_fingerprint=
            dataset_fingerprint,
        git_commit="abcdef1234567890",
        experiment_group="information_only",
        features=features,
        target=TargetSpec(
            name="btc_direction",
            horizon_seconds=900,
            threshold=0.0,
        ),
        model=ModelSpec.from_mapping(
            family="logistic_regression",
            version="1.0.0",
            parameters={
                "penalty": "l2",
                "C": 1.0,
            },
        ),
        validation=WalkForwardSpec(
            n_folds=4,
            min_train_observations=1000,
            test_observations=250,
            purge_observations=15,
            embargo_observations=15,
            expanding=True,
        ),
        random_seed=42,
    )


def test_identity_is_deterministic() -> None:
    first = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    second = experiment(
        created_at=datetime(
            2026,
            2,
            1,
            tzinfo=UTC,
        )
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )

    assert (
        first.experiment_id
        == second.experiment_id
    )


def test_dataset_change_changes_identity() -> None:
    first = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        dataset_fingerprint="a" * 64,
    )

    second = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        dataset_fingerprint="b" * 64,
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


def test_feature_parameters_change_identity() -> None:
    first = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        feature_parameter=1.0,
    )

    second = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        feature_parameter=2.0,
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


def test_feature_order_is_part_of_identity() -> None:
    first = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        feature_order=(
            "ofi",
            "spread",
        ),
    )

    second = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
        feature_order=(
            "spread",
            "ofi",
        ),
    )

    assert (
        first.fingerprint
        != second.fingerprint
    )


def test_invalid_dataset_fingerprint_is_rejected() -> None:
    with pytest.raises(
        ExperimentSpecError
    ):
        experiment(
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ),
            dataset_fingerprint=
                "not-a-sha256",
        )


def test_invalid_git_commit_is_rejected() -> None:
    with pytest.raises(
        ExperimentSpecError
    ):
        ExperimentSpec(
            display_name="bad",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ),
            dataset_fingerprint=
                "a" * 64,
            git_commit="not-git",
            experiment_group="test",
            features=(
                feature(),
            ),
            target=TargetSpec(
                name="target",
                horizon_seconds=60,
            ),
            model=ModelSpec.from_mapping(
                family="model",
                version="1",
            ),
            validation=WalkForwardSpec(
                n_folds=1,
                min_train_observations=10,
                test_observations=5,
            ),
            random_seed=1,
        )


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(
        ExperimentSpecError
    ):
        ExperimentSpec(
            display_name="duplicates",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            ),
            dataset_fingerprint=
                "a" * 64,
            git_commit="abcdef1",
            experiment_group="test",
            features=(
                feature(
                    name="ofi",
                ),
                feature(
                    name="ofi",
                ),
            ),
            target=TargetSpec(
                name="target",
                horizon_seconds=60,
            ),
            model=ModelSpec.from_mapping(
                family="model",
                version="1",
            ),
            validation=WalkForwardSpec(
                n_folds=1,
                min_train_observations=10,
                test_observations=5,
            ),
            random_seed=1,
        )


def test_target_horizon_must_be_positive() -> None:
    with pytest.raises(
        ExperimentSpecError
    ):
        TargetSpec(
            name="target",
            horizon_seconds=0,
        )


def test_walk_forward_requires_valid_sizes() -> None:
    with pytest.raises(
        ExperimentSpecError
    ):
        WalkForwardSpec(
            n_folds=0,
            min_train_observations=100,
            test_observations=20,
        )


def test_json_contains_experiment_identity() -> None:
    item = experiment(
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        )
    )

    payload = item.to_dict()

    assert (
        payload["experiment_id"]
        == item.experiment_id
    )

    assert (
        payload["fingerprint"]
        == item.fingerprint
    )
