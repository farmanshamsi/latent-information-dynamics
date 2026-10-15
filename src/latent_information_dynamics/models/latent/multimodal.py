from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LatentModelOutput:
    """
    Complete forward-pass state for the multimodal
    latent information model.

    The intermediate states are exposed intentionally
    because they will later be used for modality-specific
    optimization and gradient diagnostics.
    """

    logits: Tensor

    price_state: Tensor

    flow_state: Tensor

    latent_state: Tensor


class ModalityEncoder(nn.Module):
    """
    Small nonlinear encoder for one information modality.

    No normalization is performed internally.

    Feature standardization must be estimated from the
    legally available training sample only.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Encoder input must have shape "
                "(batch, features)."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                "Unexpected encoder feature dimension: "
                f"expected {self.input_dim}, "
                f"received {x.shape[1]}."
            )

        return self.network(
            x
        )


class MultimodalLatentModel(nn.Module):
    """
    Nonlinear Price + Trade-Flow latent information model.

    Architecture
    ------------

    Price:
        R^11 -> R^16 -> R^4

    Trade flow:
        R^20 -> R^24 -> R^4

    Fusion:
        R^8 -> R^8 -> R^4

    Prediction:
        R^4 -> binary logit

    The model emits logits rather than probabilities so that
    BCEWithLogitsLoss can be used during optimization.
    """

    def __init__(
        self,
        *,
        price_input_dim: int = 11,
        flow_input_dim: int = 20,
        price_hidden_dim: int = 16,
        flow_hidden_dim: int = 24,
        modality_latent_dim: int = 4,
        fusion_hidden_dim: int = 8,
        latent_dim: int = 4,
    ) -> None:
        super().__init__()

        if fusion_hidden_dim <= 0:
            raise ValueError(
                "fusion_hidden_dim must be positive."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        self.price_encoder = (
            ModalityEncoder(
                input_dim=
                    price_input_dim,
                hidden_dim=
                    price_hidden_dim,
                latent_dim=
                    modality_latent_dim,
            )
        )

        self.flow_encoder = (
            ModalityEncoder(
                input_dim=
                    flow_input_dim,
                hidden_dim=
                    flow_hidden_dim,
                latent_dim=
                    modality_latent_dim,
            )
        )

        fusion_input_dim = (
            2
            * modality_latent_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_dim,
                fusion_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                fusion_hidden_dim,
                latent_dim,
            ),
        )

        self.prediction_head = (
            nn.Linear(
                latent_dim,
                1,
            )
        )

    def encode(
        self,
        price_x: Tensor,
        flow_x: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        price_state = (
            self.price_encoder(
                price_x
            )
        )

        flow_state = (
            self.flow_encoder(
                flow_x
            )
        )

        return (
            price_state,
            flow_state,
        )

    def fuse(
        self,
        price_state: Tensor,
        flow_state: Tensor,
    ) -> Tensor:
        if (
            price_state.shape[0]
            != flow_state.shape[0]
        ):
            raise ValueError(
                "Price and flow batch sizes differ."
            )

        joint_state = torch.cat(
            (
                price_state,
                flow_state,
            ),
            dim=1,
        )

        return self.fusion(
            joint_state
        )

    def forward_with_state(
        self,
        price_x: Tensor,
        flow_x: Tensor,
    ) -> LatentModelOutput:
        (
            price_state,
            flow_state,
        ) = self.encode(
            price_x,
            flow_x,
        )

        latent_state = self.fuse(
            price_state,
            flow_state,
        )

        logits = (
            self.prediction_head(
                latent_state
            )
            .squeeze(
                dim=-1
            )
        )

        return LatentModelOutput(
            logits=
                logits,
            price_state=
                price_state,
            flow_state=
                flow_state,
            latent_state=
                latent_state,
        )

    def forward(
        self,
        price_x: Tensor,
        flow_x: Tensor,
    ) -> Tensor:
        return (
            self.forward_with_state(
                price_x,
                flow_x,
            )
            .logits
        )

    @torch.no_grad()
    def predict_proba(
        self,
        price_x: Tensor,
        flow_x: Tensor,
    ) -> Tensor:
        logits = self.forward(
            price_x,
            flow_x,
        )

        return torch.sigmoid(
            logits
        )
