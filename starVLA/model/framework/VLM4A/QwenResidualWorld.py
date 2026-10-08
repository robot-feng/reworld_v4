"""Qwen-VL + layer-wise residual world + native StarVLA action head.

The residual world predicts what must change between the current observation
and a requested future horizon. Every world-transformer layer is then exposed
to the corresponding layer of StarVLA's flow-matching action head, preserving
the gradient path

    action loss -> action cross-attention -> world layer -> Qwen-VL.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.ResidualLayerwiseFM_ActionHeader import (
    ResidualLayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.latent_world_model import (
    ResidualWorldConfig,
    ResidualWorldModel,
)
from starVLA.model.modules.vision_encoder import get_vision_encoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenResidualWorldDefaultConfig:
    """Framework defaults merged into ``config.framework``."""

    name: str = "QwenResidualWorld"
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
            "action_context_dim": 512,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
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
                "attention_head_dim": 64,
                "dropout": 0.1,
                "final_dropout": True,
                "interleave_self_attention": True,
                "use_canonical_forward": True,
                "norm_type": "ada_norm",
                "output_dim": 512,
                "positional_embeddings": None,
            },
        }
    )
    vision_view_indices: int | list[int] | str | None = "all"
    connect_layer_index: int = -1
    inference_horizon: int = 10_000
    world_loss_weight: float = 1.0
    world_mid_loss_weight: float = 0.5
    world_rollout_loss_weight: float = 0.5
    action_loss_weight: float = 1.0


@dataclass
class ModelBatch:
    """Prepared raw inputs at the framework/model boundary."""

    vlm_images: list[list[Image.Image]]
    world_images: list[list[Image.Image]]
    instructions: list[str]
    mid_images: list[list[Image.Image]] | None = None
    future_images: list[list[Image.Image]] | None = None
    horizons_mid: Tensor | None = None
    horizons: Tensor | None = None
    actions: Tensor | None = None
    state: Tensor | None = None


@FRAMEWORK_REGISTRY.register("QwenResidualWorld")
class QwenResidualWorld(baseframework):
    """Residual-world policy with layer-wise action supervision."""

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
            raise ValueError("QwenResidualWorld requires the full StarVLA config")
        self.config = merge_framework_config(
            QwenResidualWorldDefaultConfig,
            config,
        )
        framework = self.config.framework

        self.qwen_vl_interface = (
            qwen_vl_interface
            if qwen_vl_interface is not None
            else get_vlm_model(config=self.config)
        )
        vision_config = framework.vision_encoder
        vision_selector_keys = (
            "encoder_type",
            "model_name",
            "model_id",
            "backbone_name",
            "backone_name",
            "dino_backbone",
        )
        if vision_encoder is None and (
            vision_config is None
            or not any(
                vision_config.get(key)
                for key in vision_selector_keys
            )
        ):
            raise ValueError(
                "framework.vision_encoder must explicitly select a backend "
                "supported by get_vision_encoder"
            )
        self.vision_encoder = (
            vision_encoder
            if vision_encoder is not None
            else get_vision_encoder(config=vision_config)
        )

        rw = framework.residual_world
        vlm_dim = self._vlm_hidden_dim()
        vision_dim = self._vision_hidden_dim()
        self.residual_world = ResidualWorldModel(
            ResidualWorldConfig(
                semantic_input_dim=vlm_dim,
                vision_feature_dim=vision_dim,
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

        action_cfg = framework.action_model
        dit_cfg = action_cfg.diffusion_model_cfg
        if not bool(dit_cfg.get("use_canonical_forward", True)):
            raise ValueError(
                "QwenResidualWorld requires "
                "action_model.diffusion_model_cfg.use_canonical_forward=true"
            )

        self.context_repeat = (
            2
            if bool(dit_cfg.get("interleave_self_attention", False))
            else 1
        )
        self.world_layers = int(rw.num_layers)
        action_dim = int(rw.action_context_dim)
        head_dim = int(dit_cfg.get("attention_head_dim", 64))
        if action_dim % head_dim != 0:
            raise ValueError(
                "residual_world.action_context_dim must be divisible by "
                "action attention_head_dim"
            )

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=action_dim,
            num_dit_layers=self.world_layers * self.context_repeat,
        )
        self.config.framework.action_model.diffusion_model_cfg.output_dim = (
            action_dim
        )
        self.world_to_action = nn.Sequential(
            nn.LayerNorm(int(rw.hidden_dim)),
            nn.Linear(int(rw.hidden_dim), action_dim),
        )
        self.action_model: ResidualLayerwiseFlowmatchingActionHead = (
            action_model
            if action_model is not None
            else get_action_model(config=self.config)
        )

        self.action_horizon = int(action_cfg.action_horizon)
        self.repeated_diffusion_steps = int(
            action_cfg.get("repeated_diffusion_steps", 1)
        )
        if self.repeated_diffusion_steps < 1:
            raise ValueError("repeated_diffusion_steps must be positive")
        self.connect_layer_index = int(framework.connect_layer_index)
        self.inference_horizon = int(framework.inference_horizon)
        if self.inference_horizon < 1:
            raise ValueError("inference_horizon must be positive")
        self.world_loss_weight = float(framework.world_loss_weight)
        self.world_mid_loss_weight = float(
            framework.world_mid_loss_weight
        )
        self.world_rollout_loss_weight = float(
            framework.world_rollout_loss_weight
        )
        self.action_loss_weight = float(framework.action_loss_weight)
        if min(
            self.world_loss_weight,
            self.world_mid_loss_weight,
            self.world_rollout_loss_weight,
            self.action_loss_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.world_loss_weight == 0 and self.action_loss_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.set_vision_view_indices(framework.vision_view_indices)

        # Avoid unused-parameter failures during a world-only first stage.
        if self.action_loss_weight == 0:
            for module in (self.world_to_action, self.action_model):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    @property
    def core(self) -> ResidualWorldModel:
        """Compatibility alias used by existing world-evaluation utilities."""

        return self.residual_world

    @property
    def device(self) -> torch.device:
        return next(self.residual_world.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.residual_world.parameters()).dtype

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
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size)
        raise AttributeError("cannot determine Qwen-VL hidden size")

    def _vision_hidden_dim(self) -> int:
        if hasattr(self.vision_encoder, "num_channels"):
            return int(self.vision_encoder.num_channels)
        config = getattr(self.vision_encoder, "config", None)
        for name in ("hidden_size", "embed_dim", "num_channels"):
            if config is not None and hasattr(config, name):
                return int(getattr(config, name))
        raise AttributeError(
            "vision encoder must expose num_channels or a compatible config field"
        )

    @staticmethod
    def _parse_view_indices(
        indices: int | Iterable[int] | str | None,
    ) -> tuple[int, ...] | None:
        if indices is None:
            return None
        if isinstance(indices, str):
            if indices.strip().lower() == "all":
                return None
            raise ValueError("vision view selection string must be 'all'")
        if isinstance(indices, bool):
            raise TypeError("vision view indices cannot be booleans")
        if isinstance(indices, Integral):
            values = (int(indices),)
        else:
            if isinstance(indices, (set, frozenset)):
                raise TypeError(
                    "vision view indices must have a deterministic order"
                )
            values = tuple(indices)
            if any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in values
            ):
                raise TypeError("every vision view index must be an integer")
            values = tuple(int(index) for index in values)
        if not values:
            raise ValueError("vision view indices cannot be empty")
        if any(index < 0 for index in values):
            raise ValueError("vision view indices must be non-negative")
        if len(values) != len(set(values)):
            raise ValueError("vision view indices must be unique")
        return values

    def set_vision_view_indices(
        self,
        indices: int | Iterable[int] | str | None,
    ) -> "QwenResidualWorld":
        self.vision_view_indices = self._parse_view_indices(indices)
        self.vision_view_index = (
            None
            if self.vision_view_indices is None
            else self.vision_view_indices[0]
            if len(self.vision_view_indices) == 1
            else self.vision_view_indices
        )
        return self

    @staticmethod
    def _require(examples: Sequence[dict], *keys: str) -> None:
        missing = sorted(
            {
                key
                for key in keys
                if any(key not in sample for sample in examples)
            }
        )
        if missing:
            raise KeyError("missing required fields: " + ", ".join(missing))

    @staticmethod
    def _as_views(value: Any) -> list[Image.Image]:
        converted = to_pil_preserve(value)
        return (
            list(converted)
            if isinstance(converted, (list, tuple))
            else [converted]
        )

    def _select_views(
        self,
        batch_views: list[list[Image.Image]],
    ) -> list[list[Image.Image]]:
        selected = []
        for views in batch_views:
            indices = (
                tuple(range(len(views)))
                if self.vision_view_indices is None
                else self.vision_view_indices
            )
            invalid = [index for index in indices if index >= len(views)]
            if invalid:
                raise IndexError(
                    f"vision view indices {invalid} are invalid for a sample "
                    f"with {len(views)} views"
                )
            selected.append([views[index] for index in indices])
        if len({len(views) for views in selected}) != 1:
            raise ValueError(
                "every sample must expose the same number of selected views"
            )
        return selected

    def _tensor_field(
        self,
        examples: Sequence[dict],
        key: str,
    ) -> Tensor | None:
        present = [key in sample for sample in examples]
        if not any(present):
            return None
        if not all(present):
            raise KeyError(f"{key} must be present in every sample or none")
        values = []
        for sample in examples:
            value = np.asarray(sample[key])
            if not value.flags.writeable:
                value = value.copy()
            values.append(value)
        return torch.as_tensor(
            np.stack(values),
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _ensure_state_sequence(state: Tensor | None) -> Tensor | None:
        """Normalize raw robot state to the action head's token sequence.

        Environment adapters naturally provide one state vector per sample
        (``[B, D]``), while LeRobot training data includes a one-step temporal
        axis (``[B, 1, D]``).  The action head consumes sequence tokens, so the
        framework boundary accepts both representations and canonicalizes the
        former without leaking this model-specific shape rule into clients.
        """

        if state is None:
            return None
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                "state must be [B,state_dim] or [B,T,state_dim], "
                f"got {tuple(state.shape)}"
            )
        return state

    def _trajectory_targets(
        self,
        examples: Sequence[dict],
    ) -> tuple[
        list[list[Image.Image]],
        list[list[Image.Image]],
        Tensor,
        Tensor,
    ]:
        """Read the generic time-major trajectory field for two-step forcing."""

        midpoint_images = []
        future_images = []
        midpoint_horizons = []
        future_horizons = []
        for sample in examples:
            trajectory = sample["trajectory"]
            if not isinstance(trajectory, dict):
                raise TypeError("trajectory must be a dictionary")
            images = trajectory.get("images")
            indices = trajectory.get("observation_indices")
            if not isinstance(images, (list, tuple)) or len(images) != 3:
                raise ValueError(
                    "self-forced training requires three trajectory images "
                    "at [0, midpoint, future]"
                )
            if not isinstance(indices, (list, tuple)) or len(indices) != 3:
                raise ValueError(
                    "trajectory.observation_indices must be [0, midpoint, future]"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in indices
            ):
                raise TypeError("trajectory observation indices must be integers")
            current, midpoint, future = (int(value) for value in indices)
            if current != 0 or not 0 < midpoint < future:
                raise ValueError(
                    "trajectory observation indices must satisfy 0 < midpoint < future"
                )
            midpoint_images.append(self._as_views(images[1]))
            future_images.append(self._as_views(images[2]))
            midpoint_horizons.append(midpoint)
            future_horizons.append(future)

        return (
            self._select_views(midpoint_images),
            self._select_views(future_images),
            torch.as_tensor(
                midpoint_horizons,
                device=self.device,
                dtype=torch.long,
            ),
            torch.as_tensor(
                future_horizons,
                device=self.device,
                dtype=torch.long,
            ),
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

    def _prepare_batch(
        self,
        examples: list[dict] | dict,
        *,
        training: bool,
    ) -> ModelBatch:
        examples = [examples] if isinstance(examples, dict) else examples
        if not examples:
            raise ValueError("examples cannot be empty")

        required = ["image", "lang"]
        if training:
            required.append("trajectory")
            if self.action_loss_weight > 0:
                required.append("action")
        self._require(examples, *required)

        vlm_images = [
            self._as_views(sample["image"])
            for sample in examples
        ]
        world_images = self._select_views(vlm_images)
        if training:
            (
                mid_images,
                future_images,
                horizons_mid,
                horizons,
            ) = self._trajectory_targets(examples)
        else:
            mid_images = future_images = None
            horizons_mid = horizons = None

        image_size = self._resize_target(
            getattr(self.config.datasets.vla_data, "obs_image_size", None)
        )
        if image_size is not None:
            vlm_images = resize_images(vlm_images, target_size=image_size)
            world_images = resize_images(
                world_images,
                target_size=image_size,
            )
            if mid_images is not None:
                mid_images = resize_images(
                    mid_images,
                    target_size=image_size,
                )
            if future_images is not None:
                future_images = resize_images(
                    future_images,
                    target_size=image_size,
                )

        return ModelBatch(
            vlm_images=vlm_images,
            world_images=world_images,
            mid_images=mid_images,
            future_images=future_images,
            instructions=[str(sample["lang"]) for sample in examples],
            horizons_mid=horizons_mid,
            horizons=horizons,
            actions=(
                self._tensor_field(examples, "action")
                if training
                else None
            ),
            state=self._ensure_state_sequence(
                self._tensor_field(examples, "state")
            ),
        )

    def _encode_vlm(
        self,
        images: list[list[Image.Image]],
        instructions: list[str],
    ) -> tuple[Tensor, Tensor | None]:
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
        )
        grad_context = (
            torch.no_grad()
            if self._is_frozen(self.qwen_vl_interface)
            else nullcontext()
        )
        with grad_context:
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                output = self.qwen_vl_interface(
                    **inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        tokens = output.hidden_states[self.connect_layer_index]
        mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
        return (
            tokens.to(device=self.device, dtype=self.dtype),
            None
            if mask is None
            else mask.to(device=self.device, dtype=torch.bool),
        )

    def _encode_vision(
        self,
        images: list[list[Image.Image]],
    ) -> Tensor:
        batch_size = len(images)
        view_count = len(images[0])
        grad_context = (
            torch.no_grad()
            if self._is_frozen(self.vision_encoder)
            else nullcontext()
        )
        autocast_enabled = (
            self.device.type == "cuda"
            and self.dtype in {torch.float16, torch.bfloat16}
        )
        with grad_context:
            with torch.autocast(
                self.device.type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                pixels = self.vision_encoder.prepare_input(images)
                output = self.vision_encoder(pixels, return_dict=True)
        features = (
            output.feature_map
            if hasattr(output, "feature_map")
            else output.get("feature_map")
        )
        if (
            features is None
            or features.ndim != 4
            or min(features.shape[-2:]) <= 1
        ):
            raise ValueError(
                "vision encoder must return feature_map [B*V,C,H,W]"
            )
        if features.shape[0] != batch_size * view_count:
            raise ValueError(
                "vision encoder output batch does not match "
                "batch_size * view_count"
            )
        features = features.reshape(
            batch_size,
            view_count,
            *features.shape[1:],
        )
        features = features[:, 0] if view_count == 1 else features
        return features.to(device=self.device, dtype=self.dtype)

    def _action_targets(self, actions: Tensor) -> Tensor:
        action_dim = int(self.config.framework.action_model.action_dim)
        if actions.ndim != 3 or actions.shape[-1] != action_dim:
            raise ValueError("actions must be [B,T,action_dim]")
        if actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"action sequence must contain at least "
                f"{self.action_horizon} steps"
            )
        return actions[:, -self.action_horizon :]

    def _action_contexts(
        self,
        world_hidden_states: tuple[Tensor, ...],
    ) -> tuple[list[Tensor], Tensor]:
        if len(world_hidden_states) != self.world_layers:
            raise ValueError(
                "residual world returned an unexpected number of hidden states"
            )

        world_contexts = [
            self.world_to_action(state)
            for state in world_hidden_states
        ]
        contexts = [
            context
            for context in world_contexts
            for _ in range(self.context_repeat)
        ]
        expected = len(self.action_model.model.transformer_blocks)
        if len(contexts) != expected:
            raise RuntimeError(
                f"expected {expected} action contexts, got {len(contexts)}"
            )

        mask = torch.ones(
            contexts[0].shape[:2],
            dtype=torch.bool,
            device=contexts[0].device,
        )
        return contexts, mask

    def _run_world(
        self,
        batch: ModelBatch,
        horizons: Tensor,
        target_features: Tensor | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        vlm_tokens, vlm_mask = self._encode_vlm(
            batch.vlm_images,
            batch.instructions,
        )
        current_features = self._encode_vision(batch.world_images)
        return self.residual_world(
            vlm_tokens,
            current_features,
            horizons,
            semantic_mask=vlm_mask,
            target_features=target_features,
        )

    def forward(
        self,
        examples: list[dict] | dict = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        batch = self._prepare_batch(examples, training=True)
        with torch.no_grad():
            target_mid_features = self._encode_vision(batch.mid_images)
            target_features = self._encode_vision(batch.future_images)

        vlm_tokens, vlm_mask = self._encode_vlm(
            batch.vlm_images,
            batch.instructions,
        )
        current_features = self._encode_vision(batch.world_images)
        output = self.residual_world(
            vlm_tokens,
            current_features,
            batch.horizons_mid,
            semantic_mask=vlm_mask,
        )
        pred_mid = output["predicted_future_features"]
        rollout = self.residual_world(
            vlm_tokens,
            pred_mid,
            batch.horizons - batch.horizons_mid,
            semantic_mask=vlm_mask,
        )
        pred_future = rollout["predicted_future_features"]

        loss_mid = F.mse_loss(pred_mid, target_mid_features.detach())
        loss_rollout = F.mse_loss(pred_future, target_features.detach())
        world_loss = (
            self.world_mid_loss_weight * loss_mid
            + self.world_rollout_loss_weight * loss_rollout
        )
        policy_loss = world_loss.new_zeros(())

        if self.action_loss_weight > 0:
            actions = self._action_targets(batch.actions)
            contexts, context_mask = self._action_contexts(
                output["world_hidden_states"]
            )
            repeat = self.repeated_diffusion_steps
            policy_loss = self.action_model(
                [
                    context.repeat(repeat, 1, 1)
                    for context in contexts
                ],
                actions.repeat(repeat, 1, 1),
                (
                    None
                    if batch.state is None
                    else batch.state.repeat(repeat, 1, 1)
                ),
                encoder_attention_mask=context_mask.repeat(repeat, 1),
            )

        total_loss = (
            self.world_loss_weight * world_loss
            + self.action_loss_weight * policy_loss
        )
        return {
            # Compatibility key consumed by the standard StarVLA trainer.
            "action_loss": total_loss,
            "world_loss": world_loss.detach(),
            "world_mid_loss": loss_mid.detach(),
            "world_rollout_loss": loss_rollout.detach(),
            "policy_loss": policy_loss.detach(),
        }

    def _inference_horizons(
        self,
        batch_size: int,
        horizon: int | Iterable[int] | Tensor | None,
    ) -> Tensor:
        horizon = self.inference_horizon if horizon is None else horizon
        if isinstance(horizon, Tensor):
            values = horizon
        elif isinstance(horizon, Integral) and not isinstance(horizon, bool):
            values = torch.full(
                (batch_size,),
                int(horizon),
                dtype=torch.long,
            )
        else:
            values = torch.as_tensor(list(horizon))
        if values.is_floating_point() or values.shape != (batch_size,):
            raise ValueError(
                f"horizon must contain {batch_size} integer values"
            )
        if torch.any(values < 1):
            raise ValueError("horizon values must be positive")
        return values.to(device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def predict_residual(
        self,
        examples: list[dict] | dict,
        horizon: int | Iterable[int] | Tensor | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        batch = self._prepare_batch(examples, training=False)
        return self._run_world(
            batch,
            self._inference_horizons(
                len(batch.instructions),
                horizon,
            ),
        )

    predict_feature_delta = predict_residual

    @torch.inference_mode()
    def evaluate_world(
        self,
        examples: list[dict] | dict,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        batch = self._prepare_batch(examples, training=True)
        target = self._encode_vision(batch.future_images)
        return self._run_world(
            batch,
            batch.horizons,
            target_features=target,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict] | dict = None,
        horizon: int | Iterable[int] | Tensor | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        batch = self._prepare_batch(examples, training=False)
        output = self._run_world(
            batch,
            self._inference_horizons(
                len(batch.instructions),
                horizon,
            ),
        )
        contexts, context_mask = self._action_contexts(
            output["world_hidden_states"]
        )
        actions = self.action_model.predict_action(
            contexts,
            batch.state,
            encoder_attention_mask=context_mask,
        )
        return {
            "normalized_actions": actions.detach().float().cpu().numpy(),
            "task_change_map": (
                output["task_change_map"].detach().float().cpu().numpy()
            ),
        }


def _mock_smoke_test() -> None:
    """Exercise the full framework boundary without pretrained checkpoints."""

    from types import SimpleNamespace

    import torch.nn.functional as F
    from omegaconf import OmegaConf

    class _MockVLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(32, 24)
            self.model = SimpleNamespace(
                config=SimpleNamespace(hidden_size=24)
            )

        def build_qwenvl_inputs(self, images, instructions):
            del images
            batch_size = len(instructions)
            device = self.embedding.weight.device
            return {
                "input_ids": torch.arange(
                    6,
                    device=device,
                ).repeat(batch_size, 1),
                "attention_mask": torch.ones(
                    batch_size,
                    6,
                    dtype=torch.long,
                    device=device,
                ),
            }

        def forward(self, input_ids, **_):
            hidden = self.embedding(input_ids)
            return SimpleNamespace(hidden_states=(hidden, hidden))

    class _MockVisionEncoder(nn.Module):
        num_channels = 12
        image_size = 16

        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def prepare_input(self, images):
            tensors = [
                torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy() / 255.0
                ).permute(2, 0, 1)
                for views in images
                for image in views
            ]
            return torch.stack(tensors).to(self.scale.device)

        def forward(self, pixels, return_dict=True):
            del return_dict
            features = F.adaptive_avg_pool2d(
                pixels.mean(1, keepdim=True),
                (4, 4),
            )
            return SimpleNamespace(
                feature_map=(
                    features.repeat(1, self.num_channels, 1, 1)
                    * self.scale
                )
            )

    class _MockActionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.action_horizon = 5
            self.action_dim = 7
            self.model = SimpleNamespace(
                transformer_blocks=[object() for _ in range(4)]
            )

        def forward(
            self,
            contexts,
            actions,
            state=None,
            encoder_attention_mask=None,
        ):
            del state, encoder_attention_mask
            context_loss = sum(
                context.square().mean() for context in contexts
            )
            return actions.square().mean() + self.scale.square() * context_loss

        @torch.no_grad()
        def predict_action(
            self,
            contexts,
            state=None,
            encoder_attention_mask=None,
        ):
            del state, encoder_attention_mask
            batch_size = contexts[0].shape[0]
            value = contexts[-1].mean(dim=(1, 2)) * self.scale
            return value[:, None, None].expand(
                batch_size,
                self.action_horizon,
                self.action_dim,
            )

    config = OmegaConf.create(
        {
            "framework": {
                "name": "QwenResidualWorld",
                "vision_encoder": {"encoder_type": "mock"},
                "vision_view_indices": "all",
                "inference_horizon": 10_000,
                "residual_world": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "head_dim": 8,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "max_sequence_length": 128,
                    "absorbing_horizon": 500,
                    "action_context_dim": 32,
                },
                "action_model": {
                    "action_dim": 7,
                    "state_dim": 8,
                    "action_horizon": 5,
                    "repeated_diffusion_steps": 2,
                    "diffusion_model_cfg": {
                        "attention_head_dim": 8,
                        "interleave_self_attention": True,
                        "use_canonical_forward": True,
                    },
                },
            },
            "datasets": {"vla_data": {}},
        }
    )
    model = QwenResidualWorld(
        config,
        qwen_vl_interface=_MockVLM(),
        vision_encoder=_MockVisionEncoder(),
        action_model=_MockActionModel(),
    )

    current = Image.fromarray(
        np.full((16, 16, 3), 32, dtype=np.uint8)
    )
    wrist = Image.fromarray(
        np.full((16, 16, 3), 96, dtype=np.uint8)
    )
    future = Image.fromarray(
        np.full((16, 16, 3), 160, dtype=np.uint8)
    )
    wrist_future = Image.fromarray(
        np.full((16, 16, 3), 224, dtype=np.uint8)
    )
    sample = {
        "image": [current, wrist],
        "trajectory": {
            "images": [
                [current, wrist],
                [future, wrist_future],
                [future, wrist_future],
            ],
            "observation_indices": [0, 20, 45],
        },
        "lang": "put the red block in the box",
        "action": np.zeros((5, 7), dtype=np.float32),
        "state": np.zeros((1, 8), dtype=np.float32),
    }

    training_output = model([sample])
    assert training_output["action_loss"].ndim == 0
    training_output["action_loss"].backward()
    assert model.qwen_vl_interface.embedding.weight.grad is not None
    assert model.residual_world.blocks[0].feedback_gate.grad is not None

    at_absorbing = model.predict_residual(sample, horizon=500)
    far_future = model.predict_residual(sample, horizon=10_000)
    assert far_future["requested_horizons"].item() == 10_000
    assert far_future["effective_horizons"].item() == 500
    torch.testing.assert_close(
        at_absorbing["world_tokens"],
        far_future["world_tokens"],
    )

    prediction = model.predict_action(
        {
            "image": sample["image"],
            "lang": sample["lang"],
            "state": sample["state"],
        }
    )
    assert prediction["normalized_actions"].shape == (1, 5, 7)
    assert prediction["task_change_map"].shape == (1, 2, 4, 4)
    print("QwenResidualWorld mock framework smoke test passed")


def _configured_smoke_test(config_yaml: str) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_yaml)
    model = QwenResidualWorld(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    action_horizon = int(config.framework.action_model.action_horizon)
    action_dim = int(config.framework.action_model.action_dim)
    state_dim = int(config.framework.action_model.state_dim)
    image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    target = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    sample = {
        "image": [image],
        "trajectory": {
            "images": [[image], [target], [target]],
            "observation_indices": [0, 3, 8],
        },
        "lang": "move to the requested future state",
        "action": np.zeros(
            (action_horizon, action_dim),
            dtype=np.float32,
        ),
        "state": np.zeros((1, state_dim), dtype=np.float32),
    }
    output = model([sample])
    prediction = model.predict_action(
        examples=[
            {
                "image": sample["image"],
                "lang": sample["lang"],
                "state": sample["state"],
            }
        ]
    )
    print(
        "QwenResidualWorld configured smoke passed: "
        f"loss={float(output['action_loss']):.6f}, "
        f"actions={prediction['normalized_actions'].shape}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Real StarVLA config; omit for a lightweight mock smoke test",
    )
    arguments = parser.parse_args()
    if arguments.config_yaml is None:
        _mock_smoke_test()
    else:
        _configured_smoke_test(arguments.config_yaml)
