import numpy as np

from latent_information_dynamics.models.latent.synthetic_benchmark import (
    FLOW_DIMENSION,
    PRICE_DIMENSION,
    SyntheticLatentConfig,
    generate_synthetic_latent_dataset,
)


def test_synthetic_dataset_dimensions():
    config = SyntheticLatentConfig(
        n_train=100,
        n_validation=20,
        n_test=30,
    )

    data = generate_synthetic_latent_dataset(
        config,
        seed=7,
    )

    assert data.price_x.shape == (
        150,
        PRICE_DIMENSION,
    )

    assert data.flow_x.shape == (
        150,
        FLOW_DIMENSION,
    )

    assert data.target.shape == (
        150,
    )


def test_synthetic_generator_is_reproducible():
    config = SyntheticLatentConfig(
        n_train=100,
        n_validation=20,
        n_test=30,
    )

    first = generate_synthetic_latent_dataset(
        config,
        seed=42,
    )

    second = generate_synthetic_latent_dataset(
        config,
        seed=42,
    )

    assert np.array_equal(
        first.price_x,
        second.price_x,
    )

    assert np.array_equal(
        first.flow_x,
        second.flow_x,
    )

    assert np.array_equal(
        first.target,
        second.target,
    )


def test_flow_signal_flips_in_shifted_test_regime():
    config = SyntheticLatentConfig(
        n_train=2000,
        n_validation=500,
        n_test=500,
        flow_test_loading_multiplier=-1.0,
    )

    data = generate_synthetic_latent_dataset(
        config,
        seed=123,
    )

    development = slice(
        0,
        2500,
    )

    train_corr = np.corrcoef(
        data.latent_state[
            development
        ],
        data.flow_x[
            development,
            0,
        ],
    )[
        0,
        1,
    ]

    test_corr = np.corrcoef(
        data.latent_state[
            data.test_slice
        ],
        data.flow_x[
            data.test_slice,
            0,
        ],
    )[
        0,
        1,
    ]

    assert train_corr > 0.0
    assert test_corr < 0.0


def test_heavy_tail_flow_has_extreme_sample_kurtosis():
    gaussian = (
        generate_synthetic_latent_dataset(
            SyntheticLatentConfig(
                n_train=6000,
                n_validation=1000,
                n_test=1000,
            ),
            seed=99,
        )
    )

    heavy = (
        generate_synthetic_latent_dataset(
            SyntheticLatentConfig(
                n_train=6000,
                n_validation=1000,
                n_test=1000,
                flow_heavy_tail_df=1.5,
            ),
            seed=99,
        )
    )

    def excess_kurtosis(
        values,
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        centered = (
            values
            - np.mean(
                values
            )
        )

        variance = np.mean(
            centered ** 2
        )

        return (
            np.mean(
                centered ** 4
            )
            / variance ** 2
            - 3.0
        )

    gaussian_kurtosis = excess_kurtosis(
        gaussian.flow_x[
            :,
            0,
        ]
    )

    heavy_kurtosis = excess_kurtosis(
        heavy.flow_x[
            :,
            0,
        ]
    )

    assert (
        heavy_kurtosis
        > gaussian_kurtosis
        + 5.0
    )
