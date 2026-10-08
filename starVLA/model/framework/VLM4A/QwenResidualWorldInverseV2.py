"""Standalone residual-world policy with an explicit inverse-dynamics boundary."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from PIL import Image
from torch import Tensor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.latent_world_model.inverse_residual_world import (
    InverseResidualWorldConfig,
    InverseResidualWorldModel,
)
from starVLA.model.modules.vision_encoder import get_vision_encoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenResidualWorldInverseV2DefaultConfig:
    """Complete defaults for the standalone inverse-dynamics framework."""

    name: str = "QwenResidualWorldInverseV2"
    qwenvl: dict = field(default_factory=dict)
    vision_encoder: dict = field(default_factory=dict)
    residual_world: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "max_sequence_length": 4096,
            "absorbing_horizon": 500,
        }
    )
    inverse_dynamics: dict = field(
        default_factory=lambda: {
            "condition_dim": 512,
            # residual_plus_current | residual_delta_delta | short_long |
            # absolute_plus_current | absolute_only
            "condition_mode": "residual_plus_current",
            "delta_only": False,  # legacy alias for residual_delta_delta
            "teacher_forcing_ratio": 0.5,
            "long_inference_horizon": 80,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 8,
            "action_horizon": 8,
            "repeated_diffusion_steps": 4,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 512,
                "dropout": 0.1,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 8,
                "output_dim": 768,
                "positional_embeddings": None,
            },
        }
    )
    vision_view_indices: int | list[int] | str | None = "all"
    connect_layer_index: int = -1
    inference_horizon: int = 10_000
    world_loss_weight: float = 0.25
    world_mid_loss_weight: float = 0.5
    world_rollout_loss_weight: float = 0.5
    action_loss_weight: float = 1.0


@dataclass
class InverseBatch:
    vlm_images: list[list[Image.Image]]
    world_images: list[list[Image.Image]]
    instructions: list[str]
    mid_images: list[list[Image.Image]] | None = None
    future_images: list[list[Image.Image]] | None = None
    horizons_mid: Tensor | None = None
    horizons: Tensor | None = None
    actions: Tensor | None = None
    state: Tensor | None = None


CONDITION_MODES = (
    "residual_plus_current",  # Pred(Δ) + C-RADIO(O_t): (O_t, Δ)
    "residual_delta_delta",  # Pred(Δ) only: (Δ, Δ)
    "short_long",  # dual-horizon: (Δ_short, Δ_long)
    "absolute_plus_current",  # Pred(O_{t+h}) + C-RADIO(O_t): (O_t, F_{t+h})
    "absolute_only",  # Pred(O_{t+h}) only: (F_{t+h}, F_{t+h})
)


class TaskDeltaTokenizer(nn.Module):
    """Project a pair of vision tensors into GR00T cross-attention tokens.

    Input width is always ``2 * feature_dim``; only the semantic meaning of the
    two halves changes across condition modes A/B/C/D.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        if min(feature_dim, condition_dim, max_sequence_length) < 1:
            raise ValueError("tokenizer dimensions must be positive")
        self.feature_dim = feature_dim
        self.max_sequence_length = max_sequence_length
        self.projection = nn.Linear(2 * feature_dim, condition_dim)
        self.position = SinusoidalPositionalEmbedding(condition_dim, max_seq_length=max_sequence_length)
        self.output_norm = nn.LayerNorm(condition_dim, eps=1e-6)

    def _flatten(self, features: Tensor) -> Tensor:
        if features.ndim == 5:
            batch, views, channels, height, width = features.shape
            if channels != self.feature_dim:
                raise ValueError(f"expected {self.feature_dim} feature channels, got {channels}")
            return features.permute(0, 1, 3, 4, 2).reshape(batch, views * height * width, channels)
        if features.ndim == 4:
            if features.shape[1] != self.feature_dim:
                raise ValueError(f"expected {self.feature_dim} feature channels, got {features.shape[1]}")
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3 and features.shape[-1] == self.feature_dim:
            return features
        raise ValueError("features must be [B,V,C,H,W], [B,C,H,W], or [B,N,C]")

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("paired conditioner inputs must have identical shapes")
        tokens = self.projection(torch.cat((self._flatten(left), self._flatten(right)), dim=-1))
        if tokens.shape[1] > self.max_sequence_length:
            raise ValueError(f"condition token count exceeds {self.max_sequence_length}")
        return self.output_norm(tokens + self.position(torch.zeros_like(tokens)))

    @torch.no_grad()
    def migrate_from_duplicated_delta(self) -> None:
        """Map old ``cat(Δ, Δ)`` weights to ``cat(Δ_short, Δ_long)`` init.

        Old effective map was ``W_Σ = W1 + W2``. Set ``W_short = W_Σ`` and
        ``W_long = 0`` so behaviour starts as short-only.
        """

        weight = self.projection.weight
        half = weight.shape[1] // 2
        w1, w2 = weight[:, :half], weight[:, half:]
        w_sum = w1 + w2
        weight[:, :half].copy_(w_sum)
        weight[:, half:].zero_()


@FRAMEWORK_REGISTRY.register("QwenResidualWorldInverseV2")
class QwenResidualWorldInverseV2(baseframework):
    """Qwen-VL -> residual world -> explicit task delta -> GR00T inverse dynamics."""

    default_config = QwenResidualWorldInverseV2DefaultConfig

    def __init__(
        self,
        config: Any | None = None,
        *,
        qwen_vl_interface: nn.Module | None = None,
        vision_encoder: nn.Module | None = None,
        action_model: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if config is None:
            raise ValueError("QwenResidualWorldInverseV2 requires the full StarVLA config")

        self.config = merge_framework_config(self.default_config, config)
        framework = self.config.framework
        self.qwen_vl_interface = (
            qwen_vl_interface if qwen_vl_interface is not None else get_vlm_model(config=self.config)
        )

        vision_config = framework.vision_encoder
        selectors = ("encoder_type", "model_name", "model_id", "backbone_name", "backone_name", "dino_backbone")
        if vision_encoder is None and not any(vision_config.get(key) for key in selectors):
            raise ValueError("framework.vision_encoder must select a supported backend")
        self.vision_encoder = vision_encoder if vision_encoder is not None else get_vision_encoder(config=vision_config)

        rw = framework.residual_world
        self.residual_world = InverseResidualWorldModel(
            InverseResidualWorldConfig(
                semantic_input_dim=self._vlm_hidden_dim(),
                vision_feature_dim=self._vision_hidden_dim(),
                hidden_dim=int(rw.hidden_dim),
                num_layers=int(rw.num_layers),
                num_heads=int(rw.num_heads),
                head_dim=int(rw.head_dim),
                mlp_ratio=float(rw.mlp_ratio),
                dropout=float(rw.dropout),
                max_sequence_length=int(rw.max_sequence_length),
                absorbing_horizon=int(rw.absorbing_horizon),
            )
        )

        inverse_cfg = framework.inverse_dynamics
        condition_dim = int(inverse_cfg.condition_dim)
        mode = str(inverse_cfg.get("condition_mode", "") or "").strip()
        if not mode:
            mode = "residual_delta_delta" if bool(inverse_cfg.get("delta_only", False)) else "residual_plus_current"
        if mode not in CONDITION_MODES:
            raise ValueError(f"condition_mode must be one of {CONDITION_MODES}, got {mode!r}")
        self.condition_mode = mode
        self.delta_only = mode == "residual_delta_delta"
        self.teacher_forcing_ratio = float(inverse_cfg.get("teacher_forcing_ratio", 0.5))
        self.long_inference_horizon = int(inverse_cfg.get("long_inference_horizon", 80))
        self.action_conditioner = TaskDeltaTokenizer(
            feature_dim=self._vision_hidden_dim(),
            condition_dim=condition_dim,
            max_sequence_length=int(rw.max_sequence_length),
        )

        action_cfg = framework.action_model
        dit_cfg = action_cfg.diffusion_model_cfg
        if action_cfg.action_model_type not in {"DiT-B", "DiT-L"}:
            raise ValueError("inverse dynamics requires a standard GR00T DiT-B or DiT-L action head")
        if not bool(dit_cfg.get("interleave_self_attention", True)):
            raise ValueError("inverse dynamics requires interleave_self_attention=true")
        if int(dit_cfg.num_layers) < 2 or int(dit_cfg.num_layers) % 2:
            raise ValueError("inverse dynamics DiT depth must be a positive even number")
        dit_cfg.cross_attention_dim = condition_dim
        self.action_model = action_model if action_model is not None else get_action_model(config=self.config)

        self.action_horizon = int(action_cfg.action_horizon)
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 1))
        self.connect_layer_index = int(framework.connect_layer_index)
        self.inference_horizon = int(framework.inference_horizon)
        self.world_loss_weight = float(framework.world_loss_weight)
        self.world_mid_loss_weight = float(framework.world_mid_loss_weight)
        self.world_rollout_loss_weight = float(framework.world_rollout_loss_weight)
        self.action_loss_weight = float(framework.action_loss_weight)
        self._validate_weights()
        self.set_vision_view_indices(framework.vision_view_indices)

        if self.action_loss_weight == 0:
            self.action_conditioner.requires_grad_(False)
            self.action_model.requires_grad_(False)

    @property
    def core(self) -> InverseResidualWorldModel:
        return self.residual_world

    @property
    def device(self) -> torch.device:
        return next(self.residual_world.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.residual_world.parameters()).dtype

    def _validate_weights(self) -> None:
        weights = (
            self.world_loss_weight,
            self.world_mid_loss_weight,
            self.world_rollout_loss_weight,
            self.action_loss_weight,
        )
        if min(weights) < 0 or (self.world_loss_weight == self.action_loss_weight == 0):
            raise ValueError("loss weights must be non-negative and at least one branch must be active")
        if self.world_loss_weight > 0 and self.world_mid_loss_weight + self.world_rollout_loss_weight == 0:
            raise ValueError("at least one world anchor loss must be active when world loss is enabled")
        if min(self.action_horizon, self.repeated_diffusion_steps, self.inference_horizon) < 1:
            raise ValueError("action horizon, diffusion repeats, and inference horizon must be positive")
        if self.long_inference_horizon < 1:
            raise ValueError("long_inference_horizon must be positive")
        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

    @staticmethod
    def _is_frozen(module: nn.Module) -> bool:
        return not any(parameter.requires_grad for parameter in module.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self._is_frozen(self.qwen_vl_interface):
            self.qwen_vl_interface.eval()
        if self._is_frozen(self.vision_encoder):
            self.vision_encoder.eval()
        return self

    def _vlm_hidden_dim(self) -> int:
        config = self.qwen_vl_interface.model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        raise AttributeError("cannot determine Qwen-VL hidden size")

    def _vision_hidden_dim(self) -> int:
        if hasattr(self.vision_encoder, "num_channels"):
            return int(self.vision_encoder.num_channels)
        config = getattr(self.vision_encoder, "config", None)
        for name in ("hidden_size", "embed_dim", "num_channels"):
            if config is not None and hasattr(config, name):
                return int(getattr(config, name))
        raise AttributeError("vision encoder must expose num_channels or a compatible config field")

    def set_vision_view_indices(self, indices: int | Iterable[int] | str | None) -> "QwenResidualWorldInverseV2":
        if indices is None or (isinstance(indices, str) and indices.lower() == "all"):
            self.vision_view_indices = None
        elif isinstance(indices, str):
            raise ValueError("vision view selection string must be 'all'")
        elif isinstance(indices, Integral) and not isinstance(indices, bool):
            self.vision_view_indices = (int(indices),)
        else:
            self.vision_view_indices = tuple(int(index) for index in indices)
        if self.vision_view_indices is not None:
            if not self.vision_view_indices or min(self.vision_view_indices) < 0:
                raise ValueError("vision view indices must be non-empty and non-negative")
            if len(set(self.vision_view_indices)) != len(self.vision_view_indices):
                raise ValueError("vision view indices must be unique")
        return self

    @staticmethod
    def _as_views(value: Any) -> list[Image.Image]:
        images = to_pil_preserve(value)
        return list(images) if isinstance(images, (list, tuple)) else [images]

    def _select_views(self, batch_views: list[list[Image.Image]]) -> list[list[Image.Image]]:
        selected = []
        for views in batch_views:
            indices = range(len(views)) if self.vision_view_indices is None else self.vision_view_indices
            if any(index >= len(views) for index in indices):
                raise IndexError(f"vision view selection is invalid for a sample with {len(views)} views")
            selected.append([views[index] for index in indices])
        if len({len(views) for views in selected}) != 1:
            raise ValueError("every sample must expose the same number of selected views")
        return selected

    def _tensor_field(self, examples: Sequence[dict], key: str) -> Tensor | None:
        present = [key in sample for sample in examples]
        if not any(present):
            return None
        if not all(present):
            raise KeyError(f"{key} must be present in every sample or none")
        values = [np.asarray(sample[key]) for sample in examples]
        values = [value.copy() if not value.flags.writeable else value for value in values]
        return torch.as_tensor(np.stack(values), device=self.device, dtype=self.dtype)

    @staticmethod
    def _state_sequence(state: Tensor | None) -> Tensor | None:
        if state is not None and state.ndim == 2:
            state = state.unsqueeze(1)
        if state is not None and state.ndim != 3:
            raise ValueError("state must be [B,state_dim] or [B,T,state_dim]")
        return state

    def _trajectory_targets(
        self, examples: Sequence[dict]
    ) -> tuple[list[list[Image.Image]], list[list[Image.Image]], Tensor, Tensor]:
        mid_images, future_images, horizons_mid, horizons = [], [], [], []
        for sample in examples:
            trajectory = sample["trajectory"]
            images, indices = trajectory.get("images"), trajectory.get("observation_indices")
            if not isinstance(images, (list, tuple)) or len(images) != 3:
                raise ValueError("self-forced training requires trajectory images at [0, mid, future]")
            if not isinstance(indices, (list, tuple)) or len(indices) != 3:
                raise ValueError("trajectory.observation_indices must be [0, mid, future]")
            current, mid, future = indices
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
                raise TypeError("trajectory observation indices must be integers")
            if current != 0 or not 0 < mid < future:
                raise ValueError("trajectory indices must satisfy 0 < mid < future")
            mid_images.append(self._as_views(images[1]))
            future_images.append(self._as_views(images[2]))
            horizons_mid.append(int(mid))
            horizons.append(int(future))
        return (
            self._select_views(mid_images),
            self._select_views(future_images),
            torch.as_tensor(horizons_mid, device=self.device, dtype=torch.long),
            torch.as_tensor(horizons, device=self.device, dtype=torch.long),
        )

    @staticmethod
    def _resize_target(size: Any) -> tuple[int, int] | None:
        if size is None:
            return None
        if isinstance(size, Integral):
            return int(size), int(size)
        if len(size) != 2:
            raise ValueError("obs_image_size must be an int or a pair")
        return int(size[0]), int(size[1])

    def _prepare_batch(self, examples: list[dict] | dict, training: bool) -> InverseBatch:
        examples = [examples] if isinstance(examples, dict) else examples
        if not examples:
            raise ValueError("examples cannot be empty")
        required = ["image", "lang"] + (["trajectory"] if training else [])
        if training and self.action_loss_weight > 0:
            required.append("action")
        missing = sorted({key for key in required if any(key not in sample for sample in examples)})
        if missing:
            raise KeyError("missing required fields: " + ", ".join(missing))

        vlm_images = [self._as_views(sample["image"]) for sample in examples]
        world_images = self._select_views(vlm_images)
        if training:
            mid_images, future_images, horizons_mid, horizons = self._trajectory_targets(examples)
        else:
            mid_images = future_images = horizons_mid = horizons = None

        size = self._resize_target(getattr(self.config.datasets.vla_data, "obs_image_size", None))
        if size is not None:
            vlm_images = resize_images(vlm_images, target_size=size)
            world_images = resize_images(world_images, target_size=size)
            if training:
                mid_images = resize_images(mid_images, target_size=size)
                future_images = resize_images(future_images, target_size=size)

        return InverseBatch(
            vlm_images=vlm_images,
            world_images=world_images,
            instructions=[str(sample["lang"]) for sample in examples],
            mid_images=mid_images,
            future_images=future_images,
            horizons_mid=horizons_mid,
            horizons=horizons,
            actions=self._tensor_field(examples, "action") if training else None,
            state=self._state_sequence(self._tensor_field(examples, "state")),
        )

    def _encode_vlm(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> tuple[Tensor, Tensor | None]:
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        grad_context = torch.no_grad() if self._is_frozen(self.qwen_vl_interface) else nullcontext()
        with grad_context, torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            output = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        tokens = output.hidden_states[self.connect_layer_index].to(device=self.device, dtype=self.dtype)
        mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
        mask = None if mask is None else mask.to(device=self.device, dtype=torch.bool)
        return tokens, mask

    def _encode_vision(self, images: list[list[Image.Image]]) -> Tensor:
        batch_size, view_count = len(images), len(images[0])
        grad_context = torch.no_grad() if self._is_frozen(self.vision_encoder) else nullcontext()
        use_autocast = self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}
        with grad_context, torch.autocast(self.device.type, dtype=self.dtype, enabled=use_autocast):
            output = self.vision_encoder(self.vision_encoder.prepare_input(images), return_dict=True)
        features = output.feature_map if hasattr(output, "feature_map") else output.get("feature_map")
        if features is None or features.ndim != 4 or min(features.shape[-2:]) <= 1:
            raise ValueError("vision encoder must return feature_map [B*V,C,H,W]")
        if features.shape[0] != batch_size * view_count:
            raise ValueError("vision encoder output batch does not match batch_size * view_count")
        features = features.reshape(batch_size, view_count, *features.shape[1:])
        features = features[:, 0] if view_count == 1 else features
        return features.to(device=self.device, dtype=self.dtype)

    def _action_targets(self, actions: Tensor) -> Tensor:
        action_dim = int(self.config.framework.action_model.action_dim)
        if actions.ndim != 3 or actions.shape[-1] != action_dim or actions.shape[1] < self.action_horizon:
            raise ValueError("actions must be [B,T,action_dim] with T >= action_horizon")
        return actions[:, -self.action_horizon :]

    def _action_horizons(self, batch_size: int) -> Tensor:
        return torch.full((batch_size,), self.action_horizon, device=self.device, dtype=torch.long)

    def _inference_horizons(
        self, batch_size: int, horizon: int | Iterable[int] | Tensor | None
    ) -> Tensor:
        horizon = self.inference_horizon if horizon is None else horizon
        if isinstance(horizon, Tensor):
            values = horizon
        elif isinstance(horizon, Integral) and not isinstance(horizon, bool):
            values = torch.full((batch_size,), int(horizon), dtype=torch.long)
        else:
            values = torch.as_tensor(list(horizon))
        if values.is_floating_point() or values.shape != (batch_size,) or torch.any(values < 1):
            raise ValueError(f"horizon must contain {batch_size} positive integer values")
        return values.to(device=self.device, dtype=torch.long)

    def _run_world(self, batch: InverseBatch, horizons: Tensor) -> dict:
        semantic, semantic_mask = self._encode_vlm(batch.vlm_images, batch.instructions)
        current = self._encode_vision(batch.world_images)
        return self.residual_world.predict_features(semantic, current, horizons, semantic_mask=semantic_mask)

    def _run_world_dual(
        self,
        batch: InverseBatch,
        short_horizon: int | Tensor,
        long_horizon: int | Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        """Encode once, then query short/long horizons in a parallel 2B predict."""

        semantic, semantic_mask = self._encode_vlm(batch.vlm_images, batch.instructions)
        current = self._encode_vision(batch.world_images)
        prefilled = self.residual_world.prefill(semantic, current, semantic_mask)
        batch_size = current.shape[0]
        short = self._inference_horizons(batch_size, short_horizon)
        long = self._inference_horizons(batch_size, long_horizon)
        if torch.any(long <= short):
            raise ValueError("long horizon must be strictly greater than short horizon")
        output = self.residual_world.predict(
            torch.cat((current, current), dim=0),
            torch.cat((short, long), dim=0),
            torch.cat((prefilled, prefilled), dim=0),
        )
        short_delta = output["predicted_feature_delta"][:batch_size]
        long_delta = output["predicted_feature_delta"][batch_size:]
        short_view = {
            key: (value[:batch_size] if torch.is_tensor(value) and value.shape[:1] == (2 * batch_size,) else value)
            for key, value in output.items()
        }
        short_view["current_vision_features"] = current
        short_view["predicted_feature_delta"] = short_delta
        short_view["predicted_long_feature_delta"] = long_delta
        short_view["predicted_future_features"] = output["predicted_future_features"][:batch_size]
        short_view["predicted_long_future_features"] = output["predicted_future_features"][batch_size:]
        short_view["task_change_map"] = output["task_change_map"][:batch_size]
        return short_view, short_delta, long_delta

    def _oracle_mask(self, batch_size: int, device: torch.device) -> Tensor:
        oracle_count = int(batch_size * self.teacher_forcing_ratio)
        oracle = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if oracle_count > 0:
            oracle[torch.randperm(batch_size, device=device)[:oracle_count]] = True
        return oracle

    def _teacher_forced_pair(
        self,
        predicted_left: Tensor,
        predicted_right: Tensor,
        target_left: Tensor,
        target_right: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Paired teacher forcing: short/long (or left/right) share one sample mask."""

        if predicted_left.shape != target_left.shape or predicted_right.shape != target_right.shape:
            raise ValueError("predicted/target pairs must have matching shapes")
        if predicted_left.shape[0] != predicted_right.shape[0]:
            raise ValueError("paired teacher forcing requires a shared batch size")
        oracle = self._oracle_mask(predicted_left.shape[0], predicted_left.device)
        shape = (-1, *((1,) * (predicted_left.ndim - 1)))
        oracle = oracle.reshape(shape)
        left = torch.where(oracle, target_left.detach(), predicted_left)
        right = torch.where(oracle, target_right.detach(), predicted_right)
        return left, right

    def _teacher_forced_delta(self, predicted_delta: Tensor, target_delta: Tensor) -> Tensor:
        left, _ = self._teacher_forced_pair(predicted_delta, predicted_delta, target_delta, target_delta)
        return left

    def _action_condition_from_pair(self, left: Tensor, right: Tensor) -> Tensor:
        return self.action_conditioner(left, right)

    def _build_action_condition(
        self,
        *,
        current: Tensor,
        short_delta: Tensor,
        long_delta: Tensor | None = None,
        short_features: Tensor | None = None,
    ) -> Tensor:
        mode = self.condition_mode
        if mode == "residual_plus_current":
            return self._action_condition_from_pair(current, short_delta)
        if mode == "residual_delta_delta":
            return self._action_condition_from_pair(short_delta, short_delta)
        if mode == "short_long":
            if long_delta is None:
                raise ValueError("short_long mode requires long_delta")
            return self._action_condition_from_pair(short_delta, long_delta)
        if mode == "absolute_plus_current":
            if short_features is None:
                short_features = current + short_delta
            return self._action_condition_from_pair(current, short_features)
        if mode == "absolute_only":
            if short_features is None:
                short_features = current + short_delta
            return self._action_condition_from_pair(short_features, short_features)
        raise RuntimeError(f"unhandled condition_mode: {mode}")

    def _action_condition(self, current: Tensor, predicted_delta: Tensor) -> Tensor:
        """Backward-compatible helper for single-delta modes."""

        return self._build_action_condition(current=current, short_delta=predicted_delta)

    def forward(self, examples: list[dict] | dict = None, **_: Any) -> dict[str, Tensor]:
        batch = self._prepare_batch(examples, training=True)
        with torch.no_grad():
            target_mid = self._encode_vision(batch.mid_images)
            target_future = self._encode_vision(batch.future_images)

        semantic, semantic_mask = self._encode_vlm(batch.vlm_images, batch.instructions)
        current = self._encode_vision(batch.world_images)
        world_output = self.residual_world(
            semantic_tokens=semantic,
            current_features=current,
            target_mid_features=target_mid,
            target_future_features=target_future,
            horizons_mid=batch.horizons_mid,
            horizons_future=batch.horizons,
            semantic_mask=semantic_mask,
            mid_loss_weight=self.world_mid_loss_weight,
            rollout_loss_weight=self.world_rollout_loss_weight,
        )

        world_loss = world_output.loss
        action_loss = world_loss.new_zeros(())
        if self.action_loss_weight > 0:
            repeat = self.repeated_diffusion_steps
            target_short_delta = (target_mid - current).detach()
            target_long_delta = (target_future - current).detach()
            pred_short = world_output.predicted_mid_delta
            pred_long = world_output.predicted_long_delta

            if self.condition_mode == "short_long":
                short_delta, long_delta = self._teacher_forced_pair(
                    pred_short, pred_long, target_short_delta, target_long_delta
                )
                condition = self._build_action_condition(
                    current=current, short_delta=short_delta, long_delta=long_delta
                )
            elif self.condition_mode in ("absolute_plus_current", "absolute_only"):
                target_abs = target_mid.detach()
                pred_abs = world_output.predicted_mid_features
                abs_feat, _ = self._teacher_forced_pair(pred_abs, pred_abs, target_abs, target_abs)
                # Absolute future channel; absolute_plus_current also keeps O_t.
                condition = self._build_action_condition(
                    current=current,
                    short_delta=pred_short,
                    short_features=abs_feat,
                )
            else:
                action_delta = self._teacher_forced_delta(pred_short, target_short_delta)
                condition = self._build_action_condition(current=current, short_delta=action_delta)

            action_loss = self.action_model(
                condition.repeat(repeat, 1, 1),
                self._action_targets(batch.actions).repeat(repeat, 1, 1),
                None if batch.state is None else batch.state.repeat(repeat, 1, 1),
            )

        return {
            "loss": self.world_loss_weight * world_loss + self.action_loss_weight * action_loss,
            "action_loss": action_loss,
            "world_loss": world_loss,
            "world_mid_loss": world_output.mid_loss,
            "world_long_direct_loss": world_output.long_direct_loss,
            "world_rollout_loss": world_output.rollout_loss,
        }

    @torch.inference_mode()
    def predict_residual(
        self, examples: list[dict] | dict, horizon: int | Iterable[int] | Tensor | None = None
    ) -> dict:
        batch = self._prepare_batch(examples, training=False)
        return self._run_world(batch, self._inference_horizons(len(batch.instructions), horizon))

    predict_feature_delta = predict_residual

    @torch.inference_mode()
    def evaluate_world(self, examples: list[dict] | dict) -> dict:
        batch = self._prepare_batch(examples, training=True)
        target_mid = self._encode_vision(batch.mid_images)
        target_future = self._encode_vision(batch.future_images)
        semantic, semantic_mask = self._encode_vlm(batch.vlm_images, batch.instructions)
        current = self._encode_vision(batch.world_images)
        prefilled = self.residual_world.prefill(semantic, current, semantic_mask)
        mid_output, rollout_output = self.residual_world.rollout(
            current, batch.horizons_mid, batch.horizons, prefilled
        )
        direct_output = self.residual_world.predict(current, batch.horizons, prefilled)
        predicted_mid = mid_output["predicted_future_features"]
        predicted_rollout = rollout_output["predicted_future_features"]
        direct_output.update(
            target_vision_features=target_future,
            target_feature_delta=(target_future - current).detach(),
            target_mid_vision_features=target_mid,
            target_mid_feature_delta=(target_mid - current).detach(),
            predicted_mid_vision_features=predicted_mid,
            predicted_mid_feature_delta=predicted_mid - current,
            predicted_direct_future_vision_features=direct_output["predicted_future_features"],
            predicted_direct_future_feature_delta=direct_output["predicted_feature_delta"],
            predicted_rollout_vision_features=predicted_rollout,
            predicted_rollout_feature_delta=predicted_rollout - current,
        )
        return direct_output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict] | dict = None,
        horizon: int | Iterable[int] | Tensor | None = None,
        long_horizon: int | Iterable[int] | Tensor | None = None,
        condition_ablation: str | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        batch = self._prepare_batch(examples, training=False)
        batch_size = len(batch.instructions)

        if self.condition_mode == "short_long":
            short_h = self.action_horizon if horizon is None else horizon
            long_h = self.long_inference_horizon if long_horizon is None else long_horizon
            output, short_delta, long_delta = self._run_world_dual(batch, short_h, long_h)
            if condition_ablation == "short_short":
                long_delta = short_delta
            elif condition_ablation == "short_zero":
                long_delta = torch.zeros_like(short_delta)
            elif condition_ablation not in (None, "short_long"):
                raise ValueError("condition_ablation must be short_long|short_short|short_zero")
            condition = self._build_action_condition(
                current=output["current_vision_features"],
                short_delta=short_delta,
                long_delta=long_delta,
            )
        else:
            horizons = (
                self._action_horizons(batch_size)
                if horizon is None
                else self._inference_horizons(batch_size, horizon)
            )
            output = self._run_world(batch, horizons)
            short_delta = output["predicted_feature_delta"]
            condition = self._build_action_condition(
                current=output["current_vision_features"],
                short_delta=short_delta,
                short_features=output["predicted_future_features"],
            )

        actions = self.action_model.predict_action(condition, batch.state)
        return {
            "normalized_actions": actions.detach().float().cpu().numpy(),
            "task_change_map": output["task_change_map"].detach().float().cpu().numpy(),
        }


def _smoke_test(config_yaml: str) -> None:
    from omegaconf import OmegaConf

    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    config = OmegaConf.load(config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenResidualWorldInverseV2(config).to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    for module_name in str(config.trainer.get("freeze_modules", "")).split(","):
        module_name = module_name.strip()
        if module_name:
            getattr(model, module_name).requires_grad_(False)

    groups = build_param_lr_groups(model, config)
    assert {"residual_world", "action_conditioner", "action_model"} <= {group["name"] for group in groups}
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert all(parameter.numel() for parameter in parameters)

    if model.delta_only or model.condition_mode == "residual_delta_delta":
        feature_dim = model.action_conditioner.feature_dim
        current = torch.randn(1, 4, feature_dim, device=device, dtype=model.dtype)
        delta = torch.randn_like(current)
        torch.testing.assert_close(
            model._build_action_condition(current=current, short_delta=delta),
            model._build_action_condition(current=torch.randn_like(current), short_delta=delta),
        )

    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    target = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    action_cfg = config.framework.action_model
    sample = {
        "image": [image],
        "trajectory": {"images": [[image], [target], [target]], "observation_indices": [0, 3, 8]},
        "lang": "move to the requested future state",
        "action": rng.uniform(-0.5, 0.5, (action_cfg.action_horizon, action_cfg.action_dim)).astype(np.float32),
        "state": rng.uniform(-0.5, 0.5, (1, action_cfg.state_dim)).astype(np.float32),
    }
    output = model.train()([sample])
    output["action_loss"].backward(retain_graph=True)
    action_checked = {
        "residual_head": model.residual_world.residual_head.weight,
        "task_delta_projection": (
            model.action_conditioner.projection.bias if model.delta_only else model.action_conditioner.projection.weight
        ),
    }
    for name, parameter in action_checked.items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
            maximum = None if gradient is None else float(gradient.detach().abs().max())
            raise AssertionError(f"action loss did not reach {name} (max={maximum})")

    model.zero_grad(set_to_none=True)
    output["world_loss"].backward()
    gradient = model.residual_world.residual_head.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
        maximum = None if gradient is None else float(gradient.detach().abs().max())
        raise AssertionError(f"world loss did not reach residual_head (max={maximum})")

    prediction = model.eval().predict_action(
        examples={"image": sample["image"], "lang": sample["lang"], "state": sample["state"]}
    )
    assert all(value.ndim == 0 for value in output.values())
    assert prediction["normalized_actions"].shape == (1, action_cfg.action_horizon, action_cfg.action_dim)
    print(f"QwenResidualWorldInverseV2 smoke passed: total_loss={output['loss'].item():.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="examples/LIBERO_World/train_files/starvla_qwen_residual_world_inverse_v2.yaml",
    )
    arguments, _ = parser.parse_known_args()
    _smoke_test(arguments.config_yaml)
