import pytest

from latent_information_dynamics.models.binary import (
    BinaryModelError,
    ConstantProbabilityModel,
    LogisticRegressionBinary,
)


def test_constant_model_learns_event_rate() -> None:
    model = ConstantProbabilityModel()

    model.fit(
        X=(
            (0.0,),
            (0.0,),
            (0.0,),
            (0.0,),
        ),
        y=(
            1,
            1,
            1,
            0,
        ),
    )

    assert (
        model.probability_
        == pytest.approx(
            0.75
        )
    )


def test_constant_model_outputs_same_probability() -> None:
    model = ConstantProbabilityModel()

    model.fit(
        X=(
            (0.0,),
            (0.0,),
        ),
        y=(
            0,
            1,
        ),
    )

    probabilities = model.predict_proba(
        (
            (10.0,),
            (-10.0,),
        )
    )

    assert probabilities == (
        0.5,
        0.5,
    )


def test_logistic_model_learns_simple_direction() -> None:
    model = LogisticRegressionBinary(
        l2=0.1
    )

    X = (
        (-3.0,),
        (-2.0,),
        (-1.0,),
        (1.0,),
        (2.0,),
        (3.0,),
    )

    y = (
        0,
        0,
        0,
        1,
        1,
        1,
    )

    model.fit(
        X,
        y,
    )

    probabilities = model.predict_proba(
        (
            (-2.0,),
            (2.0,),
        )
    )

    assert (
        probabilities[0]
        < 0.5
    )

    assert (
        probabilities[1]
        > 0.5
    )


def test_logistic_probabilities_are_bounded() -> None:
    model = LogisticRegressionBinary(
        l2=1.0
    )

    model.fit(
        (
            (-1.0,),
            (0.0,),
            (1.0,),
            (2.0,),
        ),
        (
            0,
            0,
            1,
            1,
        ),
    )

    probabilities = model.predict_proba(
        (
            (-100.0,),
            (100.0,),
        )
    )

    for probability in probabilities:
        assert 0.0 < probability < 1.0


def test_logistic_requires_both_classes() -> None:
    model = LogisticRegressionBinary()

    with pytest.raises(
        BinaryModelError
    ):
        model.fit(
            (
                (1.0,),
                (2.0,),
            ),
            (
                1,
                1,
            ),
        )


def test_logistic_handles_constant_feature() -> None:
    model = LogisticRegressionBinary()

    model.fit(
        (
            (1.0,),
            (1.0,),
            (1.0,),
            (1.0,),
        ),
        (
            0,
            1,
            0,
            1,
        ),
    )

    probabilities = model.predict_proba(
        (
            (1.0,),
        )
    )

    assert (
        probabilities[0]
        == pytest.approx(
            0.5,
            abs=1e-6,
        )
    )
