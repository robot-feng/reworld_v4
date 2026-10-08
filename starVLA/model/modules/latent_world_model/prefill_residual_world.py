"""Semantic-prefill residual world model with a compact action bottleneck.

Qwen-VL writes semantics into dense world queries. Fixed learned action queries
then read each world layer independently at the action-chunk horizon.
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
class PrefillResidualWorldConfig:
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
    num_action_queries: int = 8


@dataclass(frozen=True)
class SemanticPrefillState:
    residual_tokens: Tensor
    action_tokens: Tensor


@dataclass(frozen=True)
class WorldTrainingOutput:
    loss: Tensor
    mid_loss: Tensor
    rollout_loss: Tensor
    action_hidden_states: tuple[Tensor, ...]


class SemanticPrefill(nn.Module):
    """Write VLM semantics into world queries and keep action queries learned."""

    def __init__(self, config: PrefillResidualWorldConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.semantic_projection = nn.Linear(config.semantic_input_dim, dim)
        self.residual_query = nn.Parameter(torch.empty(1, 1, dim))
        self.action_queries = nn.Parameter(torch.empty(1, config.num_action_queries, dim))
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
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(
        self, semantic_tokens: Tensor, residual_position: Tensor, semantic_mask: Tensor | None = None
    ) -> SemanticPrefillState:
        batch, token_count, _ = residual_position.shape
        semantic_tokens = self.semantic_projection(semantic_tokens)
        if semantic_mask is not None:
            if semantic_mask.shape != semantic_tokens.shape[:2]:
                raise ValueError("semantic_mask must match semantic token shape")
            semantic_tokens = semantic_tokens * semantic_mask[..., None]

        residual_queries = self.residual_query.expand(batch, token_count, -1) + residual_position
        residual_tokens = residual_queries + self.cross_attention(
            self.query_norm(residual_queries),
            encoder_hidden_states=self.semantic_norm(semantic_tokens),
            attention_mask=semantic_mask,
        )
        return SemanticPrefillState(residual_tokens, self.action_queries.expand(batch, -1, -1))


class ResidualBlock(nn.Module):
    """Update residual queries from current visual tokens under horizon conditioning."""

    def __init__(self, config: PrefillResidualWorldConfig) -> None:
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

    def forward(self, residual_tokens: Tensor, current_tokens: Tensor, horizon_condition: Tensor) -> Tensor:
        context = torch.cat((self.current_norm(current_tokens), self.residual_norm(residual_tokens)), dim=1)
        residual_tokens = residual_tokens + self.attention(
            self.query_norm(residual_tokens, horizon_condition), encoder_hidden_states=context
        )
        return residual_tokens + self.feed_forward(self.ffn_norm(residual_tokens, horizon_condition))


class ActionResamplerBlock(nn.Module):
    """Read one world layer from fixed queries without a query residual path."""

    def __init__(self, config: PrefillResidualWorldConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.query_norm = nn.LayerNorm(dim, eps=1e-6)
        self.world_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attention = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=config.num_heads,
            dim_head=config.head_dim,
            dropout=config.dropout,
            bias=False,
            out_bias=False,
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.feed_forward = FeedForward(
            dim,
            inner_dim=int(dim * config.mlp_ratio),
            dropout=config.dropout,
            activation_fn="gelu-approximate",
        )

    def forward(self, action_queries: Tensor, world_tokens: Tensor) -> Tensor:
        action_tokens = self.cross_attention(
            self.query_norm(action_queries), encoder_hidden_states=self.world_norm(world_tokens)
        )
        action_tokens = self.output_norm(action_tokens)
        return action_tokens + self.feed_forward(self.ffn_norm(action_tokens))


class PrefillResidualWorldModel(nn.Module):
    """Predict latent residuals and expose one compact action context per layer."""

    def __init__(self, config: PrefillResidualWorldConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()
        dim = config.hidden_dim
        self.world_projection = nn.Linear(config.vision_feature_dim, dim)
        self.sequence_position = SinusoidalPositionalEmbedding(dim, max_seq_length=config.max_sequence_length)
        self.horizon_embedding = HorizonEmbedding(dim)
        self.semantic_prefill = SemanticPrefill(config)
        self.blocks = nn.ModuleList(ResidualBlock(config) for _ in range(config.num_layers))
        self.action_resamplers = nn.ModuleList(ActionResamplerBlock(config) for _ in range(config.num_layers))
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.residual_head = nn.Linear(dim, config.vision_feature_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate_config(self) -> None:
        config = self.config
        if config.hidden_dim != config.num_heads * config.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")
        if min(config.semantic_input_dim, config.vision_feature_dim, config.num_layers, config.num_action_queries) < 1:
            raise ValueError("semantic, vision, layer, and action-query dimensions must be positive")
        if min(config.max_sequence_length, config.absorbing_horizon) < 1:
            raise ValueError("sequence length and absorbing horizon must be positive")

    def _flatten_features(self, features: Tensor) -> FeatureLayout:
        dim = self.config.vision_feature_dim
        if features.ndim == 4:
            _, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            return FeatureLayout(features.flatten(2).transpose(1, 2), (height, width), None)

        if features.ndim == 5:
            batch, views, channels, height, width = features.shape
            if channels != dim:
                raise ValueError(f"expected {dim} feature channels, got {channels}")
            tokens = features.permute(0, 1, 3, 4, 2).reshape(batch, views * height * width, channels).contiguous()
            return FeatureLayout(tokens, (height, width), views)

        if features.ndim == 3 and features.shape[-1] == dim:
            return FeatureLayout(features, None, None)
        raise ValueError("features must be [B,C,H,W], [B,V,C,H,W], or [B,N,C]")

    @staticmethod
    def _view_embedding(view_count: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        half = hidden_dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = torch.arange(view_count, device=device, dtype=torch.float32)[:, None] * frequency
        embedding = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if embedding.shape[-1] < hidden_dim:
            embedding = F.pad(embedding, (0, hidden_dim - embedding.shape[-1]))
        return embedding.to(dtype=dtype)

    def _position(self, layout: FeatureLayout, reference: Tensor) -> Tensor:
        if reference.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                "world token count exceeds max_sequence_length: "
                f"{reference.shape[1]} > {self.config.max_sequence_length}"
            )
        if layout.spatial_shape is None:
            return self.sequence_position(torch.zeros_like(reference))

        spatial = get_2d_sincos_pos_embed(
            self.config.hidden_dim, layout.spatial_shape, device=reference.device, output_type="pt"
        ).to(dtype=reference.dtype)
        if layout.view_count is None:
            return spatial[None]

        patches_per_view = layout.spatial_shape[0] * layout.spatial_shape[1]
        position = spatial.repeat(layout.view_count, 1)
        view_position = self._view_embedding(
            layout.view_count, self.config.hidden_dim, reference.device, reference.dtype
        )
        return (position + view_position.repeat_interleave(patches_per_view, dim=0))[None]

    def _encode_current(self, features: Tensor) -> tuple[Tensor, FeatureLayout, Tensor]:
        layout = self._flatten_features(features)
        tokens = self.world_projection(layout.current_tokens)
        position = self._position(layout, tokens)
        return tokens + position, layout, position

    def prefill(
        self, semantic_tokens: Tensor, current_features: Tensor, semantic_mask: Tensor | None = None
    ) -> SemanticPrefillState:
        if semantic_tokens.ndim != 3:
            raise ValueError("semantic_tokens must be [B,S,C]")
        layout = self._flatten_features(current_features)
        reference = current_features.new_zeros(
            current_features.shape[0], layout.current_tokens.shape[1], self.config.hidden_dim
        )
        position = self._position(layout, reference)
        semantic_tokens = semantic_tokens.to(device=position.device, dtype=position.dtype)
        semantic_mask = None if semantic_mask is None else semantic_mask.to(device=position.device, dtype=torch.bool)
        return self.semantic_prefill(semantic_tokens, position.expand(current_features.shape[0], -1, -1), semantic_mask)

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

        residual = residual_tokens.transpose(1, 2).reshape(-1, self.config.vision_feature_dim, height, width)
        future = future_tokens.transpose(1, 2).reshape(-1, self.config.vision_feature_dim, height, width)
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
            raise ValueError("residual must be [B,V,C,H,W], [B,C,H,W], or [B,N,C]")
        return residual.square().mean(dim=channel_dim).sqrt()

    def reason(
        self,
        current_features: Tensor,
        horizons: Tensor,
        prefill_state: SemanticPrefillState,
        return_action_states: bool = True,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        if horizons.shape != (current_features.shape[0],):
            raise ValueError("horizons must contain one value per sample")
        if horizons.is_floating_point() or horizons.dtype == torch.bool:
            raise TypeError("horizons must be an integer tensor with one value per sample")
        if torch.any(horizons < 1):
            raise ValueError("horizons must be positive")

        current_tokens, layout, _ = self._encode_current(current_features)
        residual_tokens = prefill_state.residual_tokens
        action_queries = prefill_state.action_tokens
        if residual_tokens.shape != current_tokens.shape:
            raise ValueError("prefilled residual tokens must match the current visual layout")
        if action_queries.shape[:2] != (current_features.shape[0], self.config.num_action_queries):
            raise ValueError("prefilled action queries have an invalid shape")

        requested_horizons = horizons.to(device=current_tokens.device, dtype=torch.long)
        effective_horizons = requested_horizons.clamp_max(self.config.absorbing_horizon)
        horizon_condition = self.horizon_embedding(effective_horizons).to(dtype=current_tokens.dtype)
        world_hidden_states, action_hidden_states = [], []
        for block, resampler in zip(self.blocks, self.action_resamplers):
            residual_tokens = block(residual_tokens, current_tokens, horizon_condition)
            world_hidden_states.append(residual_tokens)
            if return_action_states:
                action_hidden_states.append(resampler(action_queries, residual_tokens))

        predicted_future, predicted_residual = self._decode(residual_tokens, layout)
        return {
            "world_tokens": residual_tokens,
            "world_hidden_states": tuple(world_hidden_states),
            "action_hidden_states": tuple(action_hidden_states),
            "current_vision_features": current_features,
            "predicted_future_features": predicted_future,
            "predicted_future_vision_features": predicted_future,
            "predicted_feature_delta": predicted_residual,
            "task_change_map": self.task_change_map(predicted_residual),
            "requested_horizons": requested_horizons,
            "effective_horizons": effective_horizons,
        }

    def predict_features(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        horizons: Tensor,
        semantic_mask: Tensor | None = None,
        prefill_state: SemanticPrefillState | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        if prefill_state is None:
            prefill_state = self.prefill(semantic_tokens, current_features, semantic_mask)
        return self.reason(current_features, horizons, prefill_state)

    def rollout(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        horizons_mid: Tensor,
        horizons_future: Tensor,
        semantic_mask: Tensor | None = None,
        prefill_state: SemanticPrefillState | None = None,
    ) -> tuple[dict[str, Tensor | tuple[Tensor, ...]], dict[str, Tensor | tuple[Tensor, ...]]]:
        """Run the attached two-stage transition used by self-forced training."""
        remaining_horizons = horizons_future - horizons_mid
        if torch.any(remaining_horizons < 1):
            raise ValueError("self-forced horizons must satisfy horizon_mid < horizon_future")
        if prefill_state is None:
            prefill_state = self.prefill(semantic_tokens, current_features, semantic_mask)

        mid_output = self.reason(current_features, horizons_mid, prefill_state, return_action_states=False)
        rollout_output = self.reason(
            mid_output["predicted_future_features"], remaining_horizons, prefill_state, return_action_states=False
        )
        return mid_output, rollout_output

    def forward(
        self,
        semantic_tokens: Tensor,
        current_features: Tensor,
        target_mid_features: Tensor,
        target_future_features: Tensor,
        horizons_mid: Tensor,
        horizons_future: Tensor,
        action_horizons: Tensor,
        semantic_mask: Tensor | None = None,
        mid_loss_weight: float = 0.5,
        rollout_loss_weight: float = 0.5,
    ) -> WorldTrainingOutput:
        """Compute only the truth-anchor and self-forced path-consistency losses."""
        if min(mid_loss_weight, rollout_loss_weight) < 0:
            raise ValueError("self-forced loss weights must be non-negative")

        prefill_state = self.prefill(semantic_tokens, current_features, semantic_mask)
        mid_output, rollout_output = self.rollout(
            semantic_tokens,
            current_features,
            horizons_mid,
            horizons_future,
            semantic_mask,
            prefill_state,
        )
        action_output = self.reason(current_features, action_horizons, prefill_state)
        pred_mid = mid_output["predicted_future_features"]
        pred_future = rollout_output["predicted_future_features"]
        if pred_mid.shape != target_mid_features.shape or pred_future.shape != target_future_features.shape:
            raise ValueError("predicted and target feature shapes must match at both anchors")
        mid_loss = F.mse_loss(pred_mid.float(), target_mid_features.detach().float())
        rollout_loss = F.mse_loss(pred_future.float(), target_future_features.detach().float())
        return WorldTrainingOutput(
            loss=mid_loss_weight * mid_loss + rollout_loss_weight * rollout_loss,
            mid_loss=mid_loss,
            rollout_loss=rollout_loss,
            action_hidden_states=action_output["action_hidden_states"],
        )


def _smoke_test() -> None:
    torch.manual_seed(7)
    model = PrefillResidualWorldModel(
        PrefillResidualWorldConfig(
            semantic_input_dim=24,
            vision_feature_dim=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            head_dim=8,
            mlp_ratio=2.0,
            num_action_queries=4,
        )
    )
    with torch.no_grad():
        nn.init.normal_(model.residual_head.weight, std=0.02)

    semantic = torch.randn(2, 6, 24, requires_grad=True)
    current = torch.randn(2, 2, 12, 4, 4, requires_grad=True)
    target_mid = current.detach() + torch.randn_like(current) * 0.1
    target_future = target_mid + torch.randn_like(current) * 0.1
    output = model(
        semantic_tokens=semantic,
        current_features=current,
        target_mid_features=target_mid,
        target_future_features=target_future,
        horizons_mid=torch.tensor([8, 12]),
        horizons_future=torch.tensor([13, 21]),
        action_horizons=torch.tensor([8, 8]),
    )
    assert len(output.action_hidden_states) == 2
    assert output.action_hidden_states[0].shape == (2, 4, 32)
    torch.testing.assert_close(output.loss, 0.5 * (output.mid_loss + output.rollout_loss))
    action_loss = sum(state.square().mean() for state in output.action_hidden_states)
    (output.loss + action_loss).backward()
    assert semantic.grad is not None and current.grad is not None
    assert model.semantic_prefill.action_queries.grad is not None
    assert model.action_resamplers[0].cross_attention.to_q.weight.grad is not None

    model.eval()
    state = model.prefill(semantic.detach(), current.detach())
    at_absorbing = model.reason(current.detach(), torch.tensor([500, 500]), state)
    far_future = model.reason(current.detach(), torch.tensor([10_000, 10_000]), state)
    torch.testing.assert_close(at_absorbing["world_tokens"], far_future["world_tokens"])
    torch.testing.assert_close(at_absorbing["action_hidden_states"][-1], far_future["action_hidden_states"][-1])
    print("PrefillResidualWorldModel smoke test passed")


if __name__ == "__main__":
    _smoke_test()
