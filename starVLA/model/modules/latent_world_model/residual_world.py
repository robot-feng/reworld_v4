"""Task-conditioned residual world modeling in vision-feature space.

The model predicts the future latent residual

    delta(t, h) = F(min(t + h, t_terminal)) - F(t)

and exposes every intermediate world layer to a downstream layer-wise action
head. Future features are supervision only and never enter the transformer
context. The data sampler preserves the requested ``h`` when the target
saturates at the absorbing terminal state. Conditioning uses
``h_eff = min(h, absorbing_horizon)``, so arbitrarily long inference requests
map exactly to the learned terminal-horizon condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
    get_2d_sincos_pos_embed,
)
from torch import Tensor


@dataclass
class ResidualWorldConfig:
    """Shape and depth configuration for :class:`ResidualWorldModel`."""

    semantic_input_dim: int
    vision_feature_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_sequence_length: int = 4096
    absorbing_horizon: int = 500


@dataclass(frozen=True)
class FeatureLayout:
    """Flattened feature tokens plus enough metadata to restore their layout."""

    current_tokens: Tensor
    spatial_shape: tuple[int, int] | None
    view_count: int | None


class HorizonEmbedding(nn.Module):
    """Encode the requested discrete relative future horizon ``h``."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            frequency_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.time_mlp = TimestepEmbedding(frequency_dim, hidden_dim)

    def forward(self, horizon: Tensor) -> Tensor:
        if horizon.ndim != 1 or horizon.is_floating_point():
            raise TypeError("horizon must be an integer tensor shaped [B]")
        if torch.any(horizon < 1):
            raise ValueError("horizon values must be positive")
        dtype = next(self.time_mlp.parameters()).dtype
        return self.time_mlp(self.time_proj(horizon).to(dtype=dtype))


class HorizonLayerNorm(nn.Module):
    """LayerNorm modulated by the requested future horizon."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]


class ResidualWorldBlock(nn.Module):
    """Bidirectional semantic/world refinement with a world-only output path."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.semantic_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.feedback_world_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.world_to_semantic = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=hidden_dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
            out_bias=False,
        )
        # Start from one-way semantic -> world conditioning, then learn how much
        # visual feedback should update the semantic stream.
        self.feedback_gate = nn.Parameter(torch.zeros(()))

        self.world_attn_norm = HorizonLayerNorm(hidden_dim)
        self.semantic_context_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.world_context_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.world_attention = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=hidden_dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
            out_bias=False,
        )
        self.world_ffn_norm = HorizonLayerNorm(hidden_dim)
        self.feed_forward = FeedForward(
            hidden_dim,
            inner_dim=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation_fn="gelu-approximate",
        )

    def forward(
        self,
        semantic_tokens: Tensor,
        world_tokens: Tensor,
        horizon_condition: Tensor,
        semantic_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        semantic_delta = self.world_to_semantic(
            self.semantic_norm(semantic_tokens),
            encoder_hidden_states=self.feedback_world_norm(world_tokens),
        )
        semantic_tokens = (
            semantic_tokens
            + torch.tanh(self.feedback_gate) * semantic_delta
        )
        if semantic_mask is not None:
            semantic_tokens = semantic_tokens * semantic_mask[..., None]

        context = torch.cat(
            [
                self.semantic_context_norm(semantic_tokens),
                self.world_context_norm(world_tokens),
            ],
            dim=1,
        )
        context_mask = None
        if semantic_mask is not None:
            world_mask = torch.ones(
                world_tokens.shape[:2],
                dtype=torch.bool,
                device=world_tokens.device,
            )
            context_mask = torch.cat([semantic_mask, world_mask], dim=1)

        world_tokens = world_tokens + self.world_attention(
            self.world_attn_norm(world_tokens, horizon_condition),
            encoder_hidden_states=context,
            attention_mask=context_mask,
        )
        world_tokens = world_tokens + self.feed_forward(
            self.world_ffn_norm(world_tokens, horizon_condition)
        )
        return semantic_tokens, world_tokens


class ResidualWorldModel(nn.Module):
    """Predict visual-feature residuals and return every world hidden layer."""

    def __init__(self, config: ResidualWorldConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_dim != config.num_heads * config.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")
        if config.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if config.semantic_input_dim < 1 or config.vision_feature_dim < 1:
            raise ValueError("semantic_input_dim and vision_feature_dim must be positive")
        if config.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if config.absorbing_horizon < 1:
            raise ValueError("absorbing_horizon must be positive")

        self.semantic_projection = nn.Linear(
            config.semantic_input_dim,
            config.hidden_dim,
        )
        self.world_projection = nn.Linear(
            config.vision_feature_dim,
            config.hidden_dim,
        )
        self.sequence_position = SinusoidalPositionalEmbedding(
            config.hidden_dim,
            max_seq_length=config.max_sequence_length,
        )
        self.horizon_embedding = HorizonEmbedding(config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualWorldBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.head_dim,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.residual_head = nn.Linear(
            config.hidden_dim,
            config.vision_feature_dim,
        )
        # At initialization the safest future estimate is copy-current.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _flatten_features(self, features: Tensor) -> FeatureLayout:
        dim = self.config.vision_feature_dim
        if features.ndim == 4:
            _, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            return FeatureLayout(
                features.flatten(2).transpose(1, 2),
                (height, width),
                None,
            )

        if features.ndim == 5:
            batch, views, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            tokens = (
                features.permute(0, 1, 3, 4, 2)
                .reshape(batch, views * height * width, channels)
                .contiguous()
            )
            return FeatureLayout(tokens, (height, width), views)

        if features.ndim == 3 and features.shape[-1] == dim:
            return FeatureLayout(features, None, None)

        raise ValueError(
            "features must be [B,C,H,W], [B,V,C,H,W], or [B,N,C]"
        )

    @staticmethod
    def _view_embedding(
        view_count: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        half = hidden_dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = (
            torch.arange(view_count, device=device, dtype=torch.float32)[:, None]
            * frequency
        )
        embedding = torch.cat([angle.sin(), angle.cos()], dim=-1)
        if embedding.shape[-1] < hidden_dim:
            embedding = F.pad(
                embedding,
                (0, hidden_dim - embedding.shape[-1]),
            )
        return embedding.to(dtype=dtype)

    def _encode_world(
        self,
        features: Tensor,
    ) -> tuple[Tensor, FeatureLayout]:
        layout = self._flatten_features(features)
        if layout.current_tokens.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                "world token count exceeds max_sequence_length: "
                f"{layout.current_tokens.shape[1]} > "
                f"{self.config.max_sequence_length}"
            )
        tokens = self.world_projection(layout.current_tokens)
        if layout.spatial_shape is None:
            return self.sequence_position(tokens), layout

        spatial = get_2d_sincos_pos_embed(
            self.config.hidden_dim,
            layout.spatial_shape,
            device=tokens.device,
            output_type="pt",
        ).to(dtype=tokens.dtype)
        if layout.view_count is None:
            return tokens + spatial[None], layout

        patches_per_view = layout.spatial_shape[0] * layout.spatial_shape[1]
        position = spatial.repeat(layout.view_count, 1)
        position = position + self._view_embedding(
            layout.view_count,
            self.config.hidden_dim,
            tokens.device,
            tokens.dtype,
        ).repeat_interleave(patches_per_view, dim=0)
        return tokens + position[None], layout

    def _decode(
        self,
        world_tokens: Tensor,
        layout: FeatureLayout,
    ) -> tuple[Tensor, Tensor]:
        residual_tokens = self.residual_head(self.output_norm(world_tokens))
        future_tokens = layout.current_tokens + residual_tokens
        if layout.spatial_shape is None:
            return future_tokens, residual_tokens

        height, width = layout.spatial_shape
        if layout.view_count is not None:
            shape = (
                -1,
                layout.view_count,
                height,
                width,
                self.config.vision_feature_dim,
            )
            residual = (
                residual_tokens.reshape(shape)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            future = (
                future_tokens.reshape(shape)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            return future, residual

        residual = residual_tokens.transpose(1, 2).reshape(
            -1,
            self.config.vision_feature_dim,
            height,
            width,
        )
        future = future_tokens.transpose(1, 2).reshape(
            -1,
            self.config.vision_feature_dim,
            height,
            width,
        )
        return future, residual

    @staticmethod
    def task_change_map(residual: Tensor) -> Tensor:
        if residual.ndim == 5:
            channel_dim = 2
        elif residual.ndim == 4:
            channel_dim = 1
        elif residual.ndim == 3:
            channel_dim = -1
        else:
            raise ValueError(
                "residual must be [B,V,C,H,W], [B,C,H,W], or [B,N,C]"
            )
        return residual.square().mean(dim=channel_dim).sqrt()

    def forward(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        horizons: Tensor,
        semantic_mask: Tensor | None = None,
        target_features: Tensor | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        if semantic_tokens.ndim != 3:
            raise ValueError("semantic_tokens must be [B,S,C]")
        if horizons.shape != (current_features.shape[0],):
            raise ValueError(
                "horizons must contain one value per sample; "
                f"received {tuple(horizons.shape)} for batch "
                f"{current_features.shape[0]}"
            )
        if horizons.is_floating_point() or horizons.dtype == torch.bool:
            raise TypeError("horizons must use an integer tensor dtype")
        if torch.any(horizons < 1):
            raise ValueError("horizons must be positive")

        semantic_tokens = self.semantic_projection(semantic_tokens)
        semantic_mask = (
            None
            if semantic_mask is None
            else semantic_mask.to(
                device=semantic_tokens.device,
                dtype=torch.bool,
            )
        )
        if semantic_mask is not None:
            if semantic_mask.shape != semantic_tokens.shape[:2]:
                raise ValueError(
                    "semantic_mask must match semantic token shape "
                    f"{tuple(semantic_tokens.shape[:2])}"
                )
            semantic_tokens = semantic_tokens * semantic_mask[..., None]

        world_tokens, layout = self._encode_world(current_features)
        semantic_tokens = semantic_tokens.to(
            device=world_tokens.device,
            dtype=world_tokens.dtype,
        )
        requested_horizons = horizons.to(
            device=world_tokens.device,
            dtype=torch.long,
        )
        effective_horizons = requested_horizons.clamp_max(
            self.config.absorbing_horizon
        )
        horizon_condition = self.horizon_embedding(
            effective_horizons
        ).to(world_tokens.dtype)

        world_hidden_states = []
        for block in self.blocks:
            semantic_tokens, world_tokens = block(
                semantic_tokens,
                world_tokens,
                horizon_condition,
                semantic_mask,
            )
            world_hidden_states.append(world_tokens)

        predicted_future, predicted_residual = self._decode(
            world_tokens,
            layout,
        )
        result: dict[str, Tensor | tuple[Tensor, ...]] = {
            "semantic_tokens": semantic_tokens,
            "world_tokens": world_tokens,
            "world_hidden_states": tuple(world_hidden_states),
            "current_vision_features": current_features,
            "predicted_future_features": predicted_future,
            # Compatibility key used by LIBERO_World metrics and visualizers.
            "predicted_future_vision_features": predicted_future,
            "predicted_feature_delta": predicted_residual,
            "task_change_map": self.task_change_map(predicted_residual),
            "requested_horizons": requested_horizons,
            "effective_horizons": effective_horizons,
        }
        if target_features is not None:
            if target_features.shape != current_features.shape:
                raise ValueError(
                    "current and target feature shapes must match; received "
                    f"{tuple(current_features.shape)} and "
                    f"{tuple(target_features.shape)}"
                )
            target_features = target_features.detach()
            target_delta = (target_features - current_features).detach()
            world_loss = F.mse_loss(predicted_residual, target_delta)
            result.update(
                {
                    "target_vision_features": target_features,
                    "target_feature_delta": target_delta,
                    "world_loss": world_loss,
                    "feature_delta_loss": world_loss,
                }
            )
        return result

def _smoke_test() -> None:
    torch.manual_seed(7)
    model = ResidualWorldModel(
        ResidualWorldConfig(
            semantic_input_dim=24,
            vision_feature_dim=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            mlp_ratio=2.0,
            absorbing_horizon=500,
        )
    )
    with torch.no_grad():
        nn.init.normal_(model.residual_head.weight, std=0.02)

    semantic = torch.randn(2, 6, 24, requires_grad=True)
    current = torch.randn(2, 2, 12, 4, 4, requires_grad=True)
    target = current.detach() + torch.randn_like(current) * 0.1
    output = model(
        semantic,
        current,
        torch.tensor([8, 500]),
        target_features=target,
    )
    assert output["predicted_feature_delta"].shape == current.shape
    assert len(output["world_hidden_states"]) == 2
    output["world_loss"].backward()
    assert semantic.grad is not None
    assert current.grad is not None

    model.eval()
    at_absorbing = model(
        semantic.detach(),
        current.detach(),
        torch.tensor([500, 500]),
    )
    far_future = model(
        semantic.detach(),
        current.detach(),
        torch.tensor([10_000, 10_000]),
    )
    torch.testing.assert_close(
        at_absorbing["world_tokens"],
        far_future["world_tokens"],
    )
    assert far_future["effective_horizons"].tolist() == [500, 500]
    print("ResidualWorldModel smoke test passed")


if __name__ == "__main__":
    _smoke_test()
