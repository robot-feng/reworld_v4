"""Qwen-VL policy with a semantic-prefill residual world model."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.latent_world_model.prefill_residual_world import (
    PrefillResidualWorldConfig,
    PrefillResidualWorldModel,
)
from starVLA.model.modules.vision_encoder import get_vision_encoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenResidualWorldPrefillDefaultConfig:
    """Framework defaults; YAML values remain authoritative."""

    name: str = "QwenResidualWorldPrefill"
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
            "num_action_queries": 8,
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
                "norm_type": "ada_norm",
                "output_dim": 512,
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
class ModelBatch:
    vlm_images: list[list[Image.Image]]
    world_images: list[list[Image.Image]]
    instructions: list[str]
    mid_images: list[list[Image.Image]] | None = None
    future_images: list[list[Image.Image]] | None = None
    horizons_mid: Tensor | None = None
    horizons: Tensor | None = None
    actions: Tensor | None = None
    state: Tensor | None = None


@FRAMEWORK_REGISTRY.register("QwenResidualWorldPrefill")
class QwenResidualWorldPrefill(baseframework):
    """Semantic prefill -> residual rollout -> compact layer-wise action context."""

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
            raise ValueError("QwenResidualWorldPrefill requires the full StarVLA config")

        self.config = merge_framework_config(QwenResidualWorldPrefillDefaultConfig, config)
        framework = self.config.framework
        self.qwen_vl_interface = (
            qwen_vl_interface if qwen_vl_interface is not None else get_vlm_model(config=self.config)
        )

        vision_config = framework.vision_encoder
        selector_keys = ("encoder_type", "model_name", "model_id", "backbone_name", "backone_name", "dino_backbone")
        if vision_encoder is None and not any(vision_config.get(key) for key in selector_keys):
            raise ValueError("framework.vision_encoder must select a backend supported by get_vision_encoder")
        self.vision_encoder = (
            vision_encoder if vision_encoder is not None else get_vision_encoder(config=vision_config)
        )

        rw = framework.residual_world
        self.residual_world = PrefillResidualWorldModel(
            PrefillResidualWorldConfig(
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
                num_action_queries=int(rw.num_action_queries),
            )
        )

        action_cfg = framework.action_model
        dit_cfg = action_cfg.diffusion_model_cfg
        if not bool(dit_cfg.get("interleave_self_attention", True)):
            raise ValueError("QwenResidualWorldPrefill requires interleave_self_attention=true")

        self.world_layers = int(rw.num_layers)
        action_context_dim = int(rw.hidden_dim)
        if action_context_dim % int(dit_cfg.get("attention_head_dim", 64)):
            raise ValueError("residual_world.hidden_dim must be divisible by action attention_head_dim")

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=action_context_dim,
            num_dit_layers=2 * self.world_layers - 1,
        )
        self.config.framework.action_model.diffusion_model_cfg.output_dim = action_context_dim
        self.action_model: LayerwiseFlowmatchingActionHead = (
            action_model if action_model is not None else get_action_model(config=self.config)
        )

        self.action_horizon = int(action_cfg.action_horizon)
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 1))
        self.connect_layer_index = int(framework.connect_layer_index)
        self.inference_horizon = int(framework.inference_horizon)
        self.world_loss_weight = float(framework.world_loss_weight)
        self.world_mid_loss_weight = float(framework.world_mid_loss_weight)
        self.world_rollout_loss_weight = float(framework.world_rollout_loss_weight)
        self.action_loss_weight = float(framework.action_loss_weight)
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

        self.set_vision_view_indices(framework.vision_view_indices)
        if self.action_loss_weight == 0:
            for parameter in self.action_model.parameters():
                parameter.requires_grad_(False)

    @property
    def core(self) -> PrefillResidualWorldModel:
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

    def set_vision_view_indices(self, indices: int | Iterable[int] | str | None) -> "QwenResidualWorldPrefill":
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

    def _prepare_batch(self, examples: list[dict] | dict, training: bool) -> ModelBatch:
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

        return ModelBatch(
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

    def _action_contexts(self, states: tuple[Tensor, ...]) -> list[Tensor | None]:
        if len(states) != self.world_layers:
            raise ValueError("residual world returned an unexpected number of action states")
        contexts: list[Tensor | None] = []
        for state in states[:-1]:
            contexts.extend((state, None))
        contexts.append(states[-1])
        if len(contexts) != len(self.action_model.model.transformer_blocks):
            raise RuntimeError("action contexts must alternate world and self-attention blocks")
        return contexts

    def _run_world(self, batch: ModelBatch, horizons: Tensor) -> dict:
        semantic, semantic_mask = self._encode_vlm(batch.vlm_images, batch.instructions)
        current = self._encode_vision(batch.world_images)
        return self.residual_world.predict_features(
            semantic,
            current,
            horizons,
            semantic_mask=semantic_mask,
        )

    def _action_horizons(self, batch_size: int) -> Tensor:
        return torch.full((batch_size,), self.action_horizon, device=self.device, dtype=torch.long)

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
            action_horizons=self._action_horizons(len(batch.instructions)),
            semantic_mask=semantic_mask,
            mid_loss_weight=self.world_mid_loss_weight,
            rollout_loss_weight=self.world_rollout_loss_weight,
        )
        world_loss = world_output.loss
        action_loss = world_loss.new_zeros(())
        if self.action_loss_weight > 0:
            repeat = self.repeated_diffusion_steps
            contexts = [
                None if context is None else context.repeat(repeat, 1, 1)
                for context in self._action_contexts(world_output.action_hidden_states)
            ]
            action_loss = self.action_model(
                contexts,
                self._action_targets(batch.actions).repeat(repeat, 1, 1),
                None if batch.state is None else batch.state.repeat(repeat, 1, 1),
            )

        total_loss = self.world_loss_weight * world_loss + self.action_loss_weight * action_loss
        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "world_loss": world_loss,
            "world_mid_loss": world_output.mid_loss,
            "world_rollout_loss": world_output.rollout_loss,
        }

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
        prefill_state = self.residual_world.prefill(semantic, current, semantic_mask)
        mid_output, rollout_output = self.residual_world.rollout(
            semantic,
            current,
            batch.horizons_mid,
            batch.horizons,
            semantic_mask,
            prefill_state,
        )
        direct_output = self.residual_world.reason(
            current, batch.horizons, prefill_state, return_action_states=False
        )
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
        **_: Any,
    ) -> dict[str, np.ndarray]:
        """Predict actions from world states aligned to the action-chunk horizon."""
        batch = self._prepare_batch(examples, training=False)
        output = self._run_world(batch, self._action_horizons(len(batch.instructions)))
        contexts = self._action_contexts(output["action_hidden_states"])
        actions = self.action_model.predict_action(contexts, batch.state)
        return {
            "normalized_actions": actions.detach().float().cpu().numpy(),
            "task_change_map": output["task_change_map"].detach().float().cpu().numpy(),
        }


def _smoke_test(config_yaml: str) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenResidualWorldPrefill(config).to(device)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    target = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    action_cfg = config.framework.action_model
    sample = {
        "image": [image],
        "trajectory": {"images": [[image], [target], [target]], "observation_indices": [0, 3, 8]},
        "lang": "move to the requested future state",
        "action": np.zeros((action_cfg.action_horizon, action_cfg.action_dim), dtype=np.float32),
        "state": np.zeros((1, action_cfg.state_dim), dtype=np.float32),
    }
    with torch.no_grad():
        output = model.train()([sample])
        prediction = model.eval().predict_action(
            examples={"image": sample["image"], "lang": sample["lang"], "state": sample["state"]}
        )
    losses = ("loss", "action_loss", "world_loss", "world_mid_loss", "world_rollout_loss")
    assert all(output[name].ndim == 0 for name in losses)
    assert prediction["normalized_actions"].shape == (1, action_cfg.action_horizon, action_cfg.action_dim)
    print(f"QwenResidualWorldPrefill smoke passed: total_loss={output['loss'].item():.6f}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="examples/LIBERO_World/train_files/starvla_qwen_residual_world_prefill.yaml",
        help="Path to a complete StarVLA training config",
    )
    arguments, _ = parser.parse_known_args()
    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    _smoke_test(arguments.config_yaml)
