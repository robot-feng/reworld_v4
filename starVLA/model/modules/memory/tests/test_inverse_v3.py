from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.QwenResidualWorldInverseV3 import QwenResidualWorldInverseV3
from starVLA.model.framework.VLM4A.QwenResidualWorldInverseV2 import QwenResidualWorldInverseV2
from starVLA.model.tools import FRAMEWORK_REGISTRY


class Vision(nn.Module):
    num_channels = 4

    def prepare_input(self, images):
        return torch.tensor([np.asarray(im).mean() / 255 for views in images for im in views]).float()

    def forward(self, x, return_dict=True):
        features = x[:, None, None, None].expand(-1, 4, 4, 4)
        return SimpleNamespace(feature_map=features)


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=6))
        self.view_counts = []
        self.scale = nn.Parameter(torch.ones(()))

    def build_qwenvl_inputs(self, images, instructions):
        self.view_counts.append([len(x) for x in images])
        return {"x": torch.tensor([np.asarray(views[0]).mean() / 255 for views in images]).float()}

    def forward(self, x, **kwargs):
        return SimpleNamespace(hidden_states=(x[:, None, None].expand(-1, 3, 6) * self.scale,))


class Action(nn.Module):
    def predict_action(self, condition, state):
        self.condition = condition
        return condition.mean((1, 2))[:, None, None].expand(-1, 8, 7)


def make_model(enabled=True, mode="residual_plus_current", joint=False):
    cfg = OmegaConf.create({"framework": {
        "name": "QwenResidualWorldInverseV3", "ttt": {"enabled": enabled, "joint_training": joint, "sequence_training": joint, "dim": 8, "grid_size": 2, "gate_init": 0.1},
        "vision_view_indices": [1],
        "inverse_dynamics": {"condition_mode": mode},
        "residual_world": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8},
    }, "datasets": {"vla_data": {"obs_image_size": None}}})
    model = QwenResidualWorldInverseV3(cfg, qwen_vl_interface=VLM(), vision_encoder=Vision(), action_model=Action())
    with torch.no_grad():
        model.residual_world.residual_head.weight.normal_(0, 0.02)
    return model


def sample(future=150):
    def views(pixel):
        return [Image.fromarray(np.full((8, 8, 3), pixel, np.uint8)) for _ in range(2)]
    images = [views(20), views(80), views(future)]
    return {"image": images[0], "lang": "move", "trajectory": {
        "images": images, "observation_indices": [0, 8, 16], "frame_indices": [100, 108, 116]}}


def test_triplet_gradients_freezing_views_and_future_causality():
    model = make_model().train()
    assert not model.residual_world.training
    assert FRAMEWORK_REGISTRY["QwenResidualWorldInverseV2"] is QwenResidualWorldInverseV2
    assert FRAMEWORK_REGISTRY["QwenResidualWorldInverseV3"] is QwenResidualWorldInverseV3
    captured = []
    original = model.step_encoded
    def capture(*args, **kwargs):
        out, state = original(*args, **kwargs)
        captured.append((out[0]["predicted_feature_delta"].detach().clone(), state.fast_weights.detach().clone()))
        return out, state
    model.step_encoded = capture
    first = model([sample()])
    assert model.qwen_vl_interface.view_counts == [[2], [2]]  # not selected world-only view
    first["loss"].backward()
    for name, p in model.named_parameters():
        if name.startswith("motion_memory."):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, name
        else:
            assert not p.requires_grad and p.grad is None, name
    second = model([sample(220)])
    for before, after in zip(captured[:2], captured[2:]):
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert first["loss"] != second["loss"]


@pytest.mark.parametrize("mode", ["residual_plus_current", "short_long", "absolute_only"])
def test_online_action_state_and_training_inference_match(mode):
    model = make_model(mode=mode).eval()
    s = sample()
    with pytest.raises(ValueError, match="step"):
        model.predict_action(s)
    first = model.predict_action(s, step=100)
    mid = {"image": s["trajectory"]["images"][1], "lang": s["lang"]}
    second = model.predict_action(mid, step=108, state=first["ttt_state"])
    assert second["normalized_actions"].shape == (1, 8, 7)
    assert second["ttt_state"].fast_weights.count_nonzero() > 0
    outputs, state = model.observe_and_predict(s, step=100, horizons=[8])
    outputs, state = model.observe_and_predict(mid, step=108, state=state, horizons=[8])
    torch.testing.assert_close(second["ttt_state"].fast_weights, state.fast_weights)
    with pytest.raises(ValueError, match="absorbing"):
        model.predict_residual(s, horizon=10000, step=0)


def test_disabled_v3_identical_to_v2_and_checkpoint_keys():
    model = make_model(False).eval()
    v2 = QwenResidualWorldInverseV2(model.config, qwen_vl_interface=VLM(), vision_encoder=Vision(), action_model=Action()).eval()
    base_state = {k: v for k, v in model.state_dict().items() if not k.startswith("motion_memory.")}
    v2.load_state_dict(base_state, strict=True)
    missing = model.load_state_dict(base_state, strict=False)
    assert all(k.startswith("motion_memory.") for k in missing.missing_keys)
    assert not missing.unexpected_keys
    torch.testing.assert_close(model.predict_residual(sample())["predicted_feature_delta"],
                               v2.predict_residual(sample(), horizon=8)["predicted_feature_delta"], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["residual_plus_current", "residual_delta_delta", "short_long",
                                  "absolute_plus_current", "absolute_only"])
def test_bf16_zero_gate_preserves_v2_action_condition_after_feedback(mode):
    model = make_model(mode=mode).to(torch.bfloat16).eval()
    with torch.no_grad():
        model.motion_memory.gate.zero_()
    s = sample()
    first = model.predict_action(s, step=100)
    mid = {"image": s["trajectory"]["images"][1], "lang": s["lang"]}
    model.predict_action(mid, step=108, state=first["ttt_state"])
    adapted_condition = model.action_model.condition.clone()
    QwenResidualWorldInverseV2.predict_action(model, mid)
    torch.testing.assert_close(adapted_condition, model.action_model.condition, atol=0, rtol=0)
    residual = model.predict_residual(mid, step=108, state=first["ttt_state"])
    base = QwenResidualWorldInverseV2.predict_residual(model, mid, horizon=8)
    torch.testing.assert_close(residual["predicted_feature_delta"], base["predicted_feature_delta"].float(), atol=0, rtol=0)
    torch.testing.assert_close(residual["predicted_future_features"], base["predicted_future_features"].float(), atol=0, rtol=0)
