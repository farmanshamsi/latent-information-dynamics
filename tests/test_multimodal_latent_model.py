import torch

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    set_research_seed,
)


def test_multimodal_latent_shapes():
    set_research_seed(
        7
    )

    model = (
        MultimodalLatentModel()
    )

    price_x = torch.randn(
        32,
        11,
    )

    flow_x = torch.randn(
        32,
        20,
    )

    output = (
        model.forward_with_state(
            price_x,
            flow_x,
        )
    )

    assert (
        output.logits.shape
        == (32,)
    )

    assert (
        output.price_state.shape
        == (32, 4)
    )

    assert (
        output.flow_state.shape
        == (32, 4)
    )

    assert (
        output.latent_state.shape
        == (32, 4)
    )


def test_probability_output_is_valid():
    set_research_seed(
        11
    )

    model = (
        MultimodalLatentModel()
    )

    probabilities = (
        model.predict_proba(
            torch.randn(
                16,
                11,
            ),
            torch.randn(
                16,
                20,
            ),
        )
    )

    assert torch.all(
        probabilities > 0.0
    )

    assert torch.all(
        probabilities < 1.0
    )


def test_wrong_price_dimension_rejected():
    model = (
        MultimodalLatentModel()
    )

    price_x = torch.randn(
        8,
        10,
    )

    flow_x = torch.randn(
        8,
        20,
    )

    try:
        model(
            price_x,
            flow_x,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected invalid feature "
            "dimension to fail."
        )


def test_initialization_is_reproducible():
    price_x = torch.tensor(
        [
            [0.1] * 11,
            [-0.2] * 11,
        ],
        dtype=torch.float32,
    )

    flow_x = torch.tensor(
        [
            [0.3] * 20,
            [-0.4] * 20,
        ],
        dtype=torch.float32,
    )

    set_research_seed(
        123
    )

    first = (
        MultimodalLatentModel()
    )

    first_logits = (
        first(
            price_x,
            flow_x,
        )
        .detach()
        .clone()
    )

    set_research_seed(
        123
    )

    second = (
        MultimodalLatentModel()
    )

    second_logits = (
        second(
            price_x,
            flow_x,
        )
        .detach()
        .clone()
    )

    assert torch.equal(
        first_logits,
        second_logits,
    )


def test_price_latent_control_shapes():
    from latent_information_dynamics.models.latent import (
        PriceLatentModel,
    )

    set_research_seed(
        29
    )

    model = PriceLatentModel()

    price_x = torch.randn(
        32,
        11,
    )

    (
        logits,
        price_state,
        latent_state,
    ) = model.forward_with_state(
        price_x
    )

    assert logits.shape == (32,)

    assert price_state.shape == (
        32,
        4,
    )

    assert latent_state.shape == (
        32,
        4,
    )


def test_price_latent_control_shapes():
    from latent_information_dynamics.models.latent import (
        PriceLatentModel,
    )

    set_research_seed(
        29
    )

    model = PriceLatentModel()

    price_x = torch.randn(
        32,
        11,
    )

    (
        logits,
        price_state,
        latent_state,
    ) = model.forward_with_state(
        price_x
    )

    assert logits.shape == (
        32,
    )

    assert price_state.shape == (
        32,
        4,
    )

    assert latent_state.shape == (
        32,
        4,
    )
