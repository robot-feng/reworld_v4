"""Residual world model for explicit inverse dynamics.

The world model predicts a horizon-conditioned visual delta. It knows nothing
about action queries or action heads; control consumes its prediction outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, get_2d_sincos_pos_embed
from torch import Tensor

from starVLA.model.modules.latent_world_model.residual_world import (
    FeatureLayout,
    HorizonEmbedding,
    HorizonLayerNorm,
)


@dataclass
class InverseResidualWorldConfig:
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
class WorldTrainingOutput:
    """World losses use absolute feature MSE (not residual MSE).

    ``mid_loss`` / ``long_direct_loss`` form the direct-anchor term; ``rollout_loss``
    is the self-forced path term. Action heads consume only the direct residuals.
    """

    loss: Tensor
    mid_loss: Tensor
    rollout_loss: Tensor
    predicted_mid_delta: Tensor
    predicted_long_delta: Tensor
    long_direct_loss: Tensor
    predicted_mid_features: Tensor
    predicted_long_features: Tensor


class SemanticPrefill(nn.Module):
    """Write task semantics into dense world queries once per observation."""

    def __init__(self, config: InverseResidualWorldConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.semantic_projection = nn.Linear(config.semantic_input_dim, dim)
        self.residual_query = nn.Parameter(torch.empty(1, 1, dim))
        self.query_norm = nn.LayerNorm(dim, eps=1e-6)
        self.semantic_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attention = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=config.num_heads,
            dim_head=config.head_dim,
            dropout=config.dropout,
            bias=False,
            out_bias=False,
        )
        nn.init.normal_(self.residual_query, std=0.02)

    def forward(
        self, semantic_tokens: Tensor, residual_position: Tensor, semantic_mask: Tensor | None = None
    ) -> Tensor:
        semantic_tokens = self.semantic_projection(semantic_tokens)
        if semantic_mask is not None:
            if semantic_mask.shape != semantic_tokens.shape[:2]:
                raise ValueError("semantic_mask must match semantic token shape")
            semantic_tokens = semantic_tokens * semantic_mask[..., None]

        batch, token_count, _ = residual_position.shape
        queries = self.residual_query.expand(batch, token_count, -1) + residual_position
        return queries + self.cross_attention(
            self.query_norm(queries),
            encoder_hidden_states=self.semantic_norm(semantic_tokens),
            attention_mask=semantic_mask,
        )


class ResidualTransitionBlock(nn.Module):
    """Update the predicted task delta from current visual tokens."""

    def __init__(self, config: InverseResidualWorldConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.query_norm = HorizonLayerNorm(dim)
        self.current_norm = nn.LayerNorm(dim, eps=1e-6)
        self.residual_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=config.num_heads,
            dim_head=config.head_dim,
            dropout=config.dropout,
            bias=False,
            out_bias=False,
        )
        self.ffn_norm = HorizonLayerNorm(dim)
        self.feed_forward = FeedForward(
            dim,
            inner_dim=int(dim * config.mlp_ratio),
            dropout=config.dropout,
            activation_fn="gelu-approximate",
        )

    def forward(self, residual: Tensor, current: Tensor, horizon: Tensor) -> Tensor:
        context = torch.cat((self.current_norm(current), self.residual_norm(residual)), dim=1)
        residual = residual + self.attention(
            self.query_norm(residual, horizon), encoder_hidden_states=context
        )
        return residual + self.feed_forward(self.ffn_norm(residual, horizon))


class InverseResidualWorldModel(nn.Module):
    """Predict visual task deltas and train them with two-anchor self-forcing."""

    def __init__(self, config: InverseResidualWorldConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()
        dim = config.hidden_dim
        self.current_projection = nn.Linear(config.vision_feature_dim, dim)
        self.sequence_position = SinusoidalPositionalEmbedding(dim, max_seq_length=config.max_sequence_length)
        self.horizon_embedding = HorizonEmbedding(dim)
        self.semantic_prefill = SemanticPrefill(config)
        self.blocks = nn.ModuleList(ResidualTransitionBlock(config) for _ in range(config.num_layers))
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.residual_head = nn.Linear(dim, config.vision_feature_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate_config(self) -> None:
        config = self.config
        if config.hidden_dim != config.num_heads * config.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")
        if min(
            config.semantic_input_dim,
            config.vision_feature_dim,
            config.num_layers,
            config.max_sequence_length,
            config.absorbing_horizon,
        ) < 1:
            raise ValueError("world dimensions, depth, sequence length, and horizon must be positive")

    def _flatten(self, features: Tensor) -> FeatureLayout:
        dim = self.config.vision_feature_dim
        if features.ndim == 5:
            batch, views, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            tokens = features.permute(0, 1, 3, 4, 2).reshape(batch, views * height * width, channels)
            return FeatureLayout(tokens.contiguous(), (height, width), views)
        if features.ndim == 4:
            _, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            return FeatureLayout(features.flatten(2).transpose(1, 2), (height, width), None)
        if features.ndim == 3 and features.shape[-1] == dim:
            return FeatureLayout(features, None, None)
        raise ValueError("features must be [B,V,C,H,W], [B,C,H,W], or [B,N,C]")

    @staticmethod
    def _view_embedding(view_count: int, dim: int, reference: Tensor) -> Tensor:
        half = dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=reference.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = torch.arange(view_count, device=reference.device, dtype=torch.float32)[:, None] * frequency
        embedding = torch.cat((angle.sin(), angle.cos()), dim=-1)
        return F.pad(embedding, (0, dim - embedding.shape[-1])).to(dtype=reference.dtype)

    def _position(self, layout: FeatureLayout, reference: Tensor) -> Tensor:
        if reference.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"world token count exceeds max_sequence_length: "
                f"{reference.shape[1]} > {self.config.max_sequence_length}"
            )
        if layout.spatial_shape is None:
            return self.sequence_position(torch.zeros_like(reference))

        spatial = get_2d_sincos_pos_embed(
            self.config.hidden_dim,
            layout.spatial_shape,
            device=reference.device,
            output_type="pt",
        ).to(dtype=reference.dtype)
        if layout.view_count is None:
            return spatial[None]

        patches = layout.spatial_shape[0] * layout.spatial_shape[1]
        views = self._view_embedding(layout.view_count, self.config.hidden_dim, reference)
        return (spatial.repeat(layout.view_count, 1) + views.repeat_interleave(patches, dim=0))[None]

    def _encode_current(self, features: Tensor) -> tuple[Tensor, FeatureLayout, Tensor]:
        layout = self._flatten(features)
        current = self.current_projection(layout.current_tokens)
        position = self._position(layout, current)
        return current + position, layout, position

    def prefill(
        self, semantic_tokens: Tensor, current_features: Tensor, semantic_mask: Tensor | None = None
    ) -> Tensor:
        if semantic_tokens.ndim != 3:
            raise ValueError("semantic_tokens must be [B,S,C]")
        layout = self._flatten(current_features)
        reference = current_features.new_zeros(
            current_features.shape[0], layout.current_tokens.shape[1], self.config.hidden_dim
        )
        position = self._position(layout, reference).expand(current_features.shape[0], -1, -1)
        semantic = semantic_tokens.to(device=position.device, dtype=position.dtype)
        mask = None if semantic_mask is None else semantic_mask.to(device=position.device, dtype=torch.bool)
        return self.semantic_prefill(semantic, position, mask)

    def _decode(self, residual_tokens: Tensor, layout: FeatureLayout) -> tuple[Tensor, Tensor]:
        residual_tokens = self.residual_head(self.output_norm(residual_tokens))
        future_tokens = layout.current_tokens + residual_tokens
        if layout.spatial_shape is None:
            return future_tokens, residual_tokens

        height, width = layout.spatial_shape
        if layout.view_count is not None:
            shape = (-1, layout.view_count, height, width, self.config.vision_feature_dim)
            residual = residual_tokens.reshape(shape).permute(0, 1, 4, 2, 3).contiguous()
            future = future_tokens.reshape(shape).permute(0, 1, 4, 2, 3).contiguous()
            return future, residual

        shape = (-1, self.config.vision_feature_dim, height, width)
        return (
            future_tokens.transpose(1, 2).reshape(shape),
            residual_tokens.transpose(1, 2).reshape(shape),
        )

    @staticmethod
    def task_change_map(residual: Tensor) -> Tensor:
        channel_dim = 2 if residual.ndim == 5 else 1 if residual.ndim == 4 else -1
        if residual.ndim not in {3, 4, 5}:
            raise ValueError("residual must be [B,V,C,H,W], [B,C,H,W], or [B,N,C]")
        return residual.square().mean(dim=channel_dim).sqrt()

    def predict(self, current_features: Tensor, horizons: Tensor, prefilled_residual: Tensor) -> dict:
        if horizons.shape != (current_features.shape[0],):
            raise ValueError("horizons must contain one value per sample")
        if horizons.is_floating_point() or horizons.dtype == torch.bool or torch.any(horizons < 1):
            raise ValueError("horizons must be positive integer values")

        current, layout, _ = self._encode_current(current_features)
        if prefilled_residual.shape != current.shape:
            raise ValueError("prefilled residual tokens must match the current visual layout")
        requested = horizons.to(device=current.device, dtype=torch.long)
        effective = requested.clamp_max(self.config.absorbing_horizon)
        horizon = self.horizon_embedding(effective).to(dtype=current.dtype)

        residual, hidden_states = prefilled_residual, []
        for block in self.blocks:
            residual = block(residual, current, horizon)
            hidden_states.append(residual)
        predicted_features, predicted_delta = self._decode(residual, layout)
        return {
            "world_tokens": residual,
            "world_hidden_states": tuple(hidden_states),
            "current_vision_features": current_features,
            "predicted_future_features": predicted_features,
            "predicted_future_vision_features": predicted_features,
            "predicted_feature_delta": predicted_delta,
            "task_change_map": self.task_change_map(predicted_delta),
            "requested_horizons": requested,
            "effective_horizons": effective,
        }

    def predict_features(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        horizons: Tensor,
        semantic_mask: Tensor | None = None,
    ) -> dict:
        prefilled = self.prefill(semantic_tokens, current_features, semantic_mask)
        return self.predict(current_features, horizons, prefilled)

    def rollout(
        self,
        current_features: Tensor,
        horizons_mid: Tensor,
        horizons_future: Tensor,
        prefilled_residual: Tensor,
    ) -> tuple[dict, dict]:
        remaining = horizons_future - horizons_mid
        if torch.any(remaining < 1):
            raise ValueError("self-forced horizons must satisfy horizon_mid < horizon_future")
        mid = self.predict(current_features, horizons_mid, prefilled_residual)
        future = self.predict(mid["predicted_future_features"], remaining, prefilled_residual)
        return mid, future

    def forward(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        target_mid_features: Tensor,
        target_future_features: Tensor,
        horizons_mid: Tensor,
        horizons_future: Tensor,
        semantic_mask: Tensor | None = None,
        mid_loss_weight: float = 0.5,
        rollout_loss_weight: float = 0.5,
        long_direct_loss_weight: float | None = None,
    ) -> WorldTrainingOutput:
        if min(mid_loss_weight, rollout_loss_weight) < 0:
            raise ValueError("self-forced loss weights must be non-negative")
        if long_direct_loss_weight is None:
            long_direct_loss_weight = mid_loss_weight
        if long_direct_loss_weight < 0:
            raise ValueError("long_direct_loss_weight must be non-negative")

        prefilled = self.prefill(semantic_tokens, current_features, semantic_mask)
        # Same-origin direct anchors in one parallel world call (batch = 2B).
        batch = current_features.shape[0]
        direct = self.predict(
            torch.cat((current_features, current_features), dim=0),
            torch.cat((horizons_mid, horizons_future), dim=0),
            torch.cat((prefilled, prefilled), dim=0),
        )
        mid_features = direct["predicted_future_features"][:batch]
        long_features = direct["predicted_future_features"][batch:]
        mid_delta = direct["predicted_feature_delta"][:batch]
        long_delta = direct["predicted_feature_delta"][batch:]

        remaining = horizons_future - horizons_mid
        if torch.any(remaining < 1):
            raise ValueError("self-forced horizons must satisfy horizon_mid < horizon_future")
        rollout = self.predict(mid_features, remaining, prefilled)
        path_features = rollout["predicted_future_features"]

        if mid_features.shape != target_mid_features.shape or long_features.shape != target_future_features.shape:
            raise ValueError("predicted and target feature shapes must match at both anchors")

        mid_loss = F.mse_loss(mid_features.float(), target_mid_features.detach().float())
        long_direct_loss = F.mse_loss(long_features.float(), target_future_features.detach().float())
        rollout_loss = F.mse_loss(path_features.float(), target_future_features.detach().float())
        # L_anchor = 0.5*(short + long) when mid/long weights equal 0.5; path is separate.
        anchor_loss = mid_loss_weight * mid_loss + long_direct_loss_weight * long_direct_loss
        return WorldTrainingOutput(
            loss=anchor_loss + rollout_loss_weight * rollout_loss,
            mid_loss=mid_loss,
            rollout_loss=rollout_loss,
            predicted_mid_delta=mid_delta,
            predicted_long_delta=long_delta,
            long_direct_loss=long_direct_loss,
            predicted_mid_features=mid_features,
            predicted_long_features=long_features,
        )


def _smoke_test() -> None:
    torch.manual_seed(7)
    model = InverseResidualWorldModel(
        InverseResidualWorldConfig(
            semantic_input_dim=24,
            vision_feature_dim=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            mlp_ratio=2.0,
        )
    )
    with torch.no_grad():
        nn.init.normal_(model.residual_head.weight, std=0.02)

    semantic = torch.randn(2, 6, 24, requires_grad=True)
    current = torch.randn(2, 2, 12, 4, 4, requires_grad=True)
    target_mid = current.detach() + torch.randn_like(current) * 0.1
    target_future = target_mid + torch.randn_like(current) * 0.1
    transition_calls = []
    hook = model.blocks[0].register_forward_hook(lambda *_: transition_calls.append(None))
    output = model(
        semantic,
        current,
        target_mid,
        target_future,
        torch.tensor([8, 12]),
        torch.tensor([13, 21]),
    )
    hook.remove()
    # Parallel direct (2B) + path rollout (B): one fire per block call.
    assert len(transition_calls) == 2
    assert output.predicted_long_delta.shape == output.predicted_mid_delta.shape
    output.predicted_mid_delta.mean().backward(retain_graph=True)
    assert model.residual_head.weight.grad is not None
    assert model.residual_head.weight.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    semantic.grad = current.grad = None
    output.loss.backward()
    assert semantic.grad is not None and current.grad is not None
    assert model.residual_head.weight.grad is not None
    assert all(parameter.numel() for parameter in model.parameters())

    model.eval()
    prefilled = model.prefill(semantic.detach(), current.detach())
    at_absorbing = model.predict(current.detach(), torch.tensor([500, 500]), prefilled)
    far_future = model.predict(current.detach(), torch.tensor([10_000, 10_000]), prefilled)
    torch.testing.assert_close(at_absorbing["predicted_feature_delta"], far_future["predicted_feature_delta"])
    print("InverseResidualWorldModel smoke test passed")


if __name__ == "__main__":
    _smoke_test()
