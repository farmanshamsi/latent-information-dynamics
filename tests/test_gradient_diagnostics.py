import math

import torch

from latent_information_dynamics.models.latent import (
    MultimodalLatentModel,
    binary_training_step,
    capture_gradient_snapshot,
    set_research_seed,
)


def test_gradient_groups_are_observed():
    set_research_seed(
        19
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

    target = torch.randint(
        low=0,
        high=2,
        size=(32,),
    ).float()

    logits = model(
        price_x,
        flow_x,
    )

    loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            logits,
            target,
        )
    )

    loss.backward()

    snapshot = (
        capture_gradient_snapshot(
            model
        )
    )

    assert snapshot.total_norm > 0.0

    assert (
        snapshot.price_encoder_norm
        > 0.0
    )

    assert (
        snapshot.flow_encoder_norm
        > 0.0
    )

    assert (
        snapshot.fusion_norm
        > 0.0
    )

    assert (
        snapshot.prediction_head_norm
        > 0.0
    )

    assert (
        snapshot.max_abs_gradient
        > 0.0
    )

    assert (
        snapshot.nonfinite_gradient_elements
        == 0
    )


def test_training_step_updates_model():
    set_research_seed(
        23
    )

    model = (
        MultimodalLatentModel()
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    price_x = torch.randn(
        24,
        11,
    )

    flow_x = torch.randn(
        24,
        20,
    )

    target = torch.randint(
        low=0,
        high=2,
        size=(24,),
    ).float()

    before = [
        parameter
        .detach()
        .clone()
        for parameter
        in model.parameters()
    ]

    result = binary_training_step(
        model=
            model,
        optimizer=
            optimizer,
        price_x=
            price_x,
        flow_x=
            flow_x,
        target=
            target,
    )

    after = [
        parameter
        .detach()
        .clone()
        for parameter
        in model.parameters()
    ]

    assert math.isfinite(
        result.loss
    )

    assert (
        result.gradients.total_norm
        > 0.0
    )

    assert (
        result
        .gradients
        .nonfinite_gradient_elements
        == 0
    )

    assert any(
        not torch.equal(
            old,
            new,
        )
        for old, new
        in zip(
            before,
            after,
            strict=True,
        )
    )
