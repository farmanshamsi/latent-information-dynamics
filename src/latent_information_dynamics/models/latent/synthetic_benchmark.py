from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PRICE_DIMENSION = 11
FLOW_DIMENSION = 20


@dataclass(frozen=True)
class SyntheticLatentConfig:
    n_train: int = 4096
    n_validation: int = 1024
    n_test: int = 1024

    latent_rho: float = 0.92
    target_strength: float = 1.10

    price_noise_scale: float = 1.20
    flow_noise_scale: float = 1.00

    flow_test_loading_multiplier: float = 1.0
    flow_test_noise_multiplier: float = 1.0

    flow_heavy_tail_df: float | None = None


@dataclass(frozen=True)
class SyntheticLatentDataset:
    price_x: np.ndarray
    flow_x: np.ndarray

    target: np.ndarray
    true_probability: np.ndarray
    latent_state: np.ndarray

    train_slice: slice
    validation_slice: slice
    test_slice: slice


def _sigmoid(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    positive = values >= 0.0

    result = np.empty_like(
        values
    )

    result[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -values[
                    positive
                ]
            )
        )
    )

    exponential = np.exp(
        values[
            ~positive
        ]
    )

    result[
        ~positive
    ] = (
        exponential
        / (
            1.0
            + exponential
        )
    )

    return result


def _noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    heavy_tail_df: float | None,
) -> np.ndarray:
    if heavy_tail_df is None:
        return rng.normal(
            size=shape
        )

    if heavy_tail_df <= 1.0:
        raise ValueError(
            "Heavy-tail degrees of freedom "
            "must exceed 1."
        )

    return rng.standard_t(
        df=heavy_tail_df,
        size=shape,
    )


def generate_synthetic_latent_dataset(
    config: SyntheticLatentConfig,
    *,
    seed: int,
) -> SyntheticLatentDataset:
    if (
        config.n_train < 2
        or config.n_validation < 2
        or config.n_test < 2
    ):
        raise ValueError(
            "All dataset blocks must "
            "contain at least two rows."
        )

    if not (
        0.0
        <= config.latent_rho
        < 1.0
    ):
        raise ValueError(
            "latent_rho must be in [0, 1)."
        )

    total = (
        config.n_train
        + config.n_validation
        + config.n_test
    )

    rng = np.random.default_rng(
        seed
    )

    latent = np.empty(
        total,
        dtype=np.float64,
    )

    latent[
        0
    ] = rng.normal()

    innovation_scale = np.sqrt(
        1.0
        - config.latent_rho ** 2
    )

    for index in range(
        1,
        total,
    ):
        latent[
            index
        ] = (
            config.latent_rho
            * latent[
                index - 1
            ]
            + innovation_scale
            * rng.normal()
        )

    true_probability = _sigmoid(
        config.target_strength
        * latent
    )

    target = (
        rng.uniform(
            size=total
        )
        < true_probability
    ).astype(
        np.float32
    )

    price_loadings = np.linspace(
        0.35,
        0.85,
        PRICE_DIMENSION,
        dtype=np.float64,
    )

    flow_loadings = np.linspace(
        0.45,
        1.05,
        FLOW_DIMENSION,
        dtype=np.float64,
    )

    price_noise = rng.normal(
        size=(
            total,
            PRICE_DIMENSION,
        )
    )

    price_x = (
        latent[
            :,
            None,
        ]
        * price_loadings[
            None,
            :,
        ]
        + config.price_noise_scale
        * price_noise
    )

    flow_noise = _noise(
        rng,
        (
            total,
            FLOW_DIMENSION,
        ),
        heavy_tail_df=
            config.flow_heavy_tail_df,
    )

    flow_multiplier = np.ones(
        total,
        dtype=np.float64,
    )

    flow_noise_multiplier = np.ones(
        total,
        dtype=np.float64,
    )

    test_start = (
        config.n_train
        + config.n_validation
    )

    flow_multiplier[
        test_start:
    ] = (
        config
        .flow_test_loading_multiplier
    )

    flow_noise_multiplier[
        test_start:
    ] = (
        config
        .flow_test_noise_multiplier
    )

    flow_x = (
        latent[
            :,
            None,
        ]
        * flow_multiplier[
            :,
            None,
        ]
        * flow_loadings[
            None,
            :,
        ]
        + config.flow_noise_scale
        * flow_noise_multiplier[
            :,
            None,
        ]
        * flow_noise
    )

    if not (
        np.all(
            np.isfinite(
                price_x
            )
        )
        and np.all(
            np.isfinite(
                flow_x
            )
        )
        and np.all(
            np.isfinite(
                latent
            )
        )
    ):
        raise RuntimeError(
            "Synthetic generator produced "
            "non-finite values."
        )

    train_end = (
        config.n_train
    )

    validation_end = (
        train_end
        + config.n_validation
    )

    return SyntheticLatentDataset(
        price_x=
            price_x.astype(
                np.float32
            ),
        flow_x=
            flow_x.astype(
                np.float32
            ),
        target=
            target,
        true_probability=
            true_probability.astype(
                np.float64
            ),
        latent_state=
            latent,
        train_slice=
            slice(
                0,
                train_end,
            ),
        validation_slice=
            slice(
                train_end,
                validation_end,
            ),
        test_slice=
            slice(
                validation_end,
                total,
            ),
    )
