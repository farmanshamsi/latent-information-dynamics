import numpy as np

from latent_information_dynamics.models.latent import (
    FeatureStandardizer,
)


def test_training_sample_is_standardized():
    training = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ]

    standardizer = (
        FeatureStandardizer.fit(
            training
        )
    )

    transformed = (
        standardizer.transform(
            training
        )
    )

    assert np.allclose(
        transformed.mean(
            axis=0
        ),
        0.0,
        atol=1e-7,
    )

    assert np.allclose(
        transformed.std(
            axis=0
        ),
        1.0,
        atol=1e-6,
    )


def test_constant_feature_is_safe():
    training = [
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ]

    standardizer = (
        FeatureStandardizer.fit(
            training
        )
    )

    transformed = (
        standardizer.transform(
            training
        )
    )

    assert (
        standardizer.scales[1]
        == 1.0
    )

    assert np.allclose(
        transformed[:, 1],
        0.0,
    )


def test_held_out_extreme_does_not_affect_fit():
    training = [
        [0.0],
        [1.0],
        [2.0],
    ]

    held_out = [
        [1000.0],
    ]

    standardizer = (
        FeatureStandardizer.fit(
            training
        )
    )

    assert np.isclose(
        standardizer.means[0],
        1.0,
    )

    expected_scale = np.std(
        np.asarray(
            training,
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    )[0]

    assert np.isclose(
        standardizer.scales[0],
        expected_scale,
    )

    transformed_test = (
        standardizer.transform(
            held_out
        )
    )

    assert (
        transformed_test[0, 0]
        > 1000.0
    )
