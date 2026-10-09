"""V2 backbone plus causal motion/error fast memory.

Joint mode combines the V2 self-forcing objective with causal sequence feedback.
Legacy memory-only mode remains available; ttt.enabled=false restores V2. Online calls require explicit frame steps and returned state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import torch
from torch.nn import functional as F

from starVLA.model.framework.VLM4A.QwenResidualWorldInverseV2 import (
    QwenResidualWorldInverseV2, QwenResidualWorldInverseV2DefaultConfig,
)
from starVLA.model.modules.memory.motion_error_memory import MotionErrorMemory
from starVLA.model.modules.memory.online_residual_ttt import (
    EpisodeState, observe_and_predict, time_vector,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenResidualWorldInverseV3DefaultConfig(QwenResidualWorldInverseV2DefaultConfig):
    name: str = "QwenResidualWorldInverseV3"
    inference_horizon: int = 8
    ttt: dict = field(default_factory=lambda: {
        "enabled": True, "dim": 128, "grid_size": 7,
        "inner_lr": 0.01, "forget_factor": 0.995,
        "gate_init": 1e-3, "value_mode": "motion_error",
    })


@FRAMEWORK_REGISTRY.register("QwenResidualWorldInverseV3")
class QwenResidualWorldInverseV3(QwenResidualWorldInverseV2):
    default_config = QwenResidualWorldInverseV3DefaultConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        cfg = self.config.framework.ttt
        self.ttt_enabled = bool(cfg.enabled)
        self.joint_training = bool(cfg.get("joint_training", False))
        self.training_step = 0
        if self.joint_training and (not self.ttt_enabled or not cfg.get("sequence_training", False)):
            raise ValueError("joint_training requires enabled sequential TTT")
        if self.joint_training:
            feedback_horizons = [int(h) for h in cfg.get("horizons", [self.action_horizon])]
            if feedback_horizons != [self.action_horizon]:
                raise ValueError(
                    "joint V3 TTT feedback must use exactly action_horizon "
                    f"({self.action_horizon}); train longer horizons in the base world-model branch"
                )
        self.motion_memory = MotionErrorMemory(
            self._vision_hidden_dim(), self._vlm_hidden_dim(),
            dim=int(cfg.dim), grid_size=int(cfg.grid_size),
            inner_lr=float(cfg.inner_lr), forget_factor=float(cfg.forget_factor),
            gate_init=float(cfg.gate_init), value_mode=str(cfg.value_mode),
        )
        if self.ttt_enabled and not self.joint_training:
            for module in self._base_modules():
                module.requires_grad_(False)
                module.eval()
        elif not self.ttt_enabled:
            self.motion_memory.requires_grad_(False)
        if self.joint_training:
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.eval()

    def _base_modules(self):
        return (self.qwen_vl_interface, self.vision_encoder, self.residual_world,
                self.action_conditioner, self.action_model)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "ttt_enabled", False) and not self.joint_training:
            for module in self._base_modules():
                module.eval()
        if getattr(self, "joint_training", False):
            self.vision_encoder.eval()
        return self

    def step_encoded(self, semantic, current, *, steps, horizons,
                     state: EpisodeState | None = None, semantic_mask=None):
        """Shared training/online step; horizons is a list of scalars or [B] tensors.

        State is caller-owned, never stored on the model. Cached-feature training
        can call this method directly and detach state at TBPTT boundaries.
        """
        if not self.ttt_enabled:
            raise ValueError("step_encoded requires ttt.enabled=true")
        limit = self.residual_world.config.absorbing_horizon
        hs = [time_vector(h, current.shape[0], current.device, positive=True) for h in horizons]
        if any(torch.any(h > limit) for h in hs):
            raise ValueError("TTT horizons must not exceed absorbing_horizon")

        def base_predict(s, z, h, semantic_mask=None):
            with torch.no_grad():
                return self.residual_world.predict_features(s, z, h, semantic_mask=semantic_mask)

        return observe_and_predict(
            self.motion_memory, base_predict, semantic=semantic.detach(), current=current.detach(),
            horizons=hs, steps=steps, state=state, semantic_mask=semantic_mask,
        )

    def _encode_observation(self, examples):
        batch = self._prepare_batch(examples, training=False)
        with torch.no_grad():
            semantic, mask = self._encode_vlm(batch.vlm_images, batch.instructions)
            current = self._encode_vision(batch.world_images)
        return batch, semantic, mask, current

    def forward_ttt(self, examples, *, return_predictions=False):
        """Triplet proof of principle: only the middle observation writes feedback.

        Future features are encoded only AFTER both causal steps have completed.
        This branch intentionally trains memory only, without recomputing V2 loss.
        """
        examples = [examples] if isinstance(examples, dict) else examples
        if not examples:
            raise ValueError("examples cannot be empty")
        if bool(self.config.framework.ttt.get("sequence_training", False)) and not return_predictions:
            from starVLA.model.modules.memory.sequence_training import forward_sequence
            return forward_sequence(self, examples)
        # Reuse V2 validation, but keep complete images for the middle Qwen input.
        _, _, mid_h, future_h = self._trajectory_targets(examples)
        starts = []
        for sample in examples:
            trajectory = sample["trajectory"]
            frames = trajectory.get("frame_indices")
            offsets = trajectory["observation_indices"]
            if frames is None:
                starts.append(0)
            else:
                frame_tensor = time_vector(frames, 3, self.device)
                if not torch.equal(frame_tensor - frame_tensor[0], torch.tensor(offsets, device=self.device)):
                    raise ValueError("frame_indices must agree with observation_indices")
                starts.append(int(frame_tensor[0]))
        starts = torch.tensor(starts, device=self.device, dtype=torch.long)
        middle = [{"image": s["trajectory"]["images"][1], "lang": s["lang"]} for s in examples]
        future = [{"image": s["trajectory"]["images"][2], "lang": s["lang"]} for s in examples]
        _, h0, mask0, z0 = self._encode_observation(examples)
        _, state = self.step_encoded(h0, z0, steps=starts, horizons=[mid_h], semantic_mask=mask0)
        _, hm, maskm, zm = self._encode_observation(middle)
        predictions, _ = self.step_encoded(hm, zm, steps=starts + mid_h,
                                           horizons=[future_h - mid_h], state=state, semantic_mask=maskm)
        output = predictions[0]
        future_batch = self._prepare_batch(future, training=False)
        with torch.no_grad():
            zf = self._encode_vision(future_batch.world_images)
        target = (zf.float() - zm.float()).detach()
        loss = F.mse_loss(output["predicted_feature_delta"].float(), target)
        base_loss = F.mse_loss(output["base_predicted_feature_delta"].float(), target)
        if return_predictions:
            return {
                "ttt_predicted_future_features": output["predicted_future_features"],
                "ttt_base_future_features": zm + output["base_predicted_feature_delta"],
                "ttt_target_future_features": zf,
            }
        return {"loss": loss, "ttt_loss": loss, "ttt_base_loss": base_loss.detach(),
                "ttt_future_gain": (base_loss - loss).detach(),
                "ttt_correction_rms": output["memory_correction"].float().square().mean().sqrt().detach()}

    def forward(self, examples=None, **kwargs):
        if not self.ttt_enabled:
            return super().forward(examples, **kwargs)
        if self.joint_training:
            return self.forward_joint(examples)
        return self.forward_ttt(examples)

    @contextmanager
    def feedback_backbone_eval(self):
        # Feedback features use deterministic current weights, without retaining
        # per-anchor VLM graphs. These modules still train through the baseline loss.
        modules = (self.qwen_vl_interface, self.vision_encoder, self.residual_world)
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)

    def joint_base_examples(self, examples):
        result = []
        for sample in examples:
            trajectory = sample["trajectory"]
            offsets = list(trajectory["observation_indices"])
            desired = [0, self.action_horizon, self.long_inference_horizon]
            if any(h not in offsets for h in desired):
                raise ValueError("joint sequence must contain both configured horizon targets")
            indices = [offsets.index(h) for h in desired]
            selected = dict(trajectory, images=[trajectory["images"][i] for i in indices],
                            observation_indices=desired)
            if "frame_indices" in trajectory:
                selected["frame_indices"] = [trajectory["frame_indices"][i] for i in indices]
            item = dict(sample, image=trajectory["images"][0], trajectory=selected,
                        action=trajectory["actions"][0])
            if "states" in trajectory:
                item["state"] = trajectory["states"][0]
            result.append(item)
        return result

    def auxiliary_loss_weight(self):
        cfg = self.config.framework.ttt
        warmup = int(cfg.get("aux_warmup_steps", 2000))
        target_weight = float(cfg.get("aux_loss_weight", 0.25))
        if warmup < 0 or target_weight < 0:
            raise ValueError("aux warmup and weight must be nonnegative")
        fraction = min(1.0, (self.training_step + 1) / max(1, warmup))
        return target_weight * fraction

    def forward_joint(self, examples):
        examples = [examples] if isinstance(examples, dict) else examples
        # Preserve the established self-forcing and action objectives. The
        # auxiliary branch trains memory + action modules in the SAME backward.
        base = super().forward(self.joint_base_examples(examples))
        with self.feedback_backbone_eval():
            feedback = self.forward_ttt(examples)
        weight = self.auxiliary_loss_weight()
        output = dict(base)
        output.update({k: v for k, v in feedback.items() if k != "loss"})
        output["ttt_aux_weight"] = base["loss"].new_tensor(weight)
        output["loss"] = base["loss"] + weight * feedback["loss"]
        return output

    @torch.no_grad()
    def observe_and_predict(self, examples, *, step, state=None, horizons=None):
        """Use actual episode frame indices, not the number of policy calls."""
        _, semantic, mask, current = self._encode_observation(examples)
        return self.step_encoded(semantic, current, steps=step,
                                 horizons=[self.inference_horizon] if horizons is None else horizons,
                                 state=state, semantic_mask=mask)

    @torch.no_grad()
    def predict_residual(self, examples, horizon=None, *, step=None, state=None):
        if not self.ttt_enabled:
            return super().predict_residual(examples, horizon)
        if step is None:
            raise ValueError("TTT prediction requires step=<actual frame index> and explicit episode state")
        outputs, next_state = self.observe_and_predict(
            examples, step=step, state=state,
            horizons=[self.inference_horizon if horizon is None else horizon])
        return dict(outputs[0], ttt_state=next_state.detach())

    predict_feature_delta = predict_residual

    @torch.no_grad()
    def predict_action(self, examples=None, horizon=None, long_horizon=None,
                       condition_ablation=None, *, step=None, state=None, **kwargs):
        if not self.ttt_enabled:
            return super().predict_action(examples, horizon, long_horizon, condition_ablation, **kwargs)
        if step is None:
            raise ValueError("TTT actions require step=<actual frame index>; pass returned ttt_state next time")
        batch, semantic, mask, current = self._encode_observation(examples)
        hs = [self.action_horizon if horizon is None else horizon]
        if self.condition_mode == "short_long":
            hs.append(self.long_inference_horizon if long_horizon is None else long_horizon)
            if torch.any(time_vector(hs[1], current.shape[0], current.device, positive=True) <=
                         time_vector(hs[0], current.shape[0], current.device, positive=True)):
                raise ValueError("long horizon must exceed short horizon")
        if bool(self.config.framework.ttt.get("sequence_training", False)):
            # Keep online feedback horizons aligned with the configured causal
            # sequence-training objective. Joint training uses only action_horizon.
            for h in self.config.framework.ttt.horizons:
                if not any(torch.all(time_vector(existing, current.shape[0], current.device, positive=True) == int(h))
                           for existing in hs):
                    hs.append(int(h))
        outputs, next_state = self.step_encoded(semantic, current, steps=step, horizons=hs,
                                                state=state, semantic_mask=mask)
        output = outputs[0]
        short = output["predicted_feature_delta"]
        long = outputs[1]["predicted_feature_delta"] if self.condition_mode == "short_long" else None
        if self.condition_mode == "short_long":
            if condition_ablation == "short_short":
                long = short
            elif condition_ablation == "short_zero":
                long = torch.zeros_like(short)
            elif condition_ablation not in (None, "short_long"):
                raise ValueError("condition_ablation must be short_long|short_short|short_zero")
        # The memory/loss path remains FP32; the frozen action conditioner uses
        # the backbone dtype. Never cast corrections before residual addition.
        condition = self._build_action_condition(
            current=current, short_delta=short.to(current.dtype),
            long_delta=None if long is None else long.to(current.dtype),
            short_features=output["predicted_future_features"].to(current.dtype),
        )
        actions = self.action_model.predict_action(condition, batch.state)
        return {"normalized_actions": actions.detach().float().cpu().numpy(),
                "task_change_map": output["task_change_map"].float().cpu().numpy(),
                "ttt_state": next_state.detach()}

    @torch.no_grad()
    def evaluate_world(self, examples):
        # Keep the LIBERO_World visualizer/metric contract; add separate feedback
        # metrics rather than conflating the different prediction origins.
        output = super().evaluate_world(examples)
        if self.ttt_enabled:
            output.update(self.forward_ttt(examples, return_predictions=True))
        return output
