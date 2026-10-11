from datetime import datetime, timedelta, timezone

from latent_information_dynamics.datasets.supervised import (
    SupervisedDataset,
    SupervisedExample,
)
from latent_information_dynamics.evaluation.walk_forward import (
    evaluate_walk_forward,
    expanding_walk_forward_folds,
)
from latent_information_dynamics.models.binary import (
    ConstantProbabilityModel,
)


UTC = timezone.utc


def example(
    index: int,
    target: int,
    *,
    label_delay_hours: int = 4,
) -> SupervisedExample:
    decision = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    ) + timedelta(
        hours=4 * index
    )

    resolution = (
        decision
        + timedelta(
            hours=4
        )
    )

    return SupervisedExample(
        decision_time=decision,
        resolution_time=resolution,
        label_available_time=(
            decision
            + timedelta(
                hours=
                    label_delay_hours
            )
        ),
        feature_names=(
            "price_feature",
        ),
        feature_values=(
            float(
                target
            ),
        ),
        target=target,
        realized_return=(
            0.01
            if target == 1
            else -0.01
        ),
        realized_log_return=(
            0.00995
            if target == 1
            else -0.01005
        ),
    )


def dataset() -> SupervisedDataset:
    targets = (
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    )

    return SupervisedDataset(
        examples=tuple(
            example(
                index,
                target,
            )
            for index, target
            in enumerate(
                targets
            )
        )
    )


def test_training_labels_are_available_before_test() -> None:
    data = dataset()

    folds = expanding_walk_forward_folds(
        data,
        min_train_observations=4,
        test_observations=2,
    )

    for fold in folds:
        for index in fold.train_indices:
            assert (
                data.examples[
                    index
                ].label_available_time
                <= fold.training_cutoff
            )


def test_test_examples_are_strictly_future() -> None:
    data = dataset()

    folds = expanding_walk_forward_folds(
        data,
        min_train_observations=4,
        test_observations=2,
    )

    for fold in folds:
        latest_train_decision = max(
            data.examples[
                index
            ].decision_time
            for index
            in fold.train_indices
        )

        earliest_test_decision = min(
            data.examples[
                index
            ].decision_time
            for index
            in fold.test_indices
        )

        assert (
            latest_train_decision
            < earliest_test_decision
        )


def test_walk_forward_produces_out_of_sample_forecasts() -> None:
    result = evaluate_walk_forward(
        dataset(),
        feature_names=(
            "price_feature",
        ),
        model_factory=
            ConstantProbabilityModel,
        min_train_observations=4,
        test_observations=2,
    )

    assert len(
        result.forecasts
    ) > 0

    assert (
        result.log_loss
        > 0
    )

    assert (
        0.0
        <= result.brier_score
        <= 1.0
    )

    assert (
        0.0
        <= result.accuracy
        <= 1.0
    )


def test_forecast_times_are_chronological() -> None:
    result = evaluate_walk_forward(
        dataset(),
        feature_names=(
            "price_feature",
        ),
        model_factory=
            ConstantProbabilityModel,
        min_train_observations=4,
        test_observations=2,
    )

    assert (
        result.decision_times
        == tuple(
            sorted(
                result.decision_times
            )
        )
    )
