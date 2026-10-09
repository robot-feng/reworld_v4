from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from examples.LIBERO_World.train_files.prepare_inverse_v3_config import prepare
from examples.LIBERO_World.train_files.ttt_trainer import TTTFeedbackEvalMixin


class LegacyTrainer:
    def eval_action_model(self, step_metrics=None):
        self.legacy_metrics = step_metrics
        return {"legacy": True}


class Trainer(TTTFeedbackEvalMixin, LegacyTrainer):
    def __init__(self, model, examples):
        self.model, self.examples = model, examples
        self.accelerator = SimpleNamespace(
            unwrap_model=lambda model: model,
            reduce=lambda tensor, reduction: tensor,
            is_main_process=True,
            wait_for_everyone=lambda: None,
        )

    def _get_next_batch(self):
        return self.examples


@pytest.mark.parametrize("training", [True, False])
def test_periodic_eval_runs_real_feedback_and_restores_mode(training, monkeypatch):
    from starVLA.model.modules.memory.tests.test_inverse_v3 import make_model, sample

    model = make_model().train(training)
    trainer = Trainer(model, [sample()])
    forward = model.forward_ttt

    def check_forward(examples):
        assert not torch.is_grad_enabled()
        assert not model.training
        return forward(examples)

    def reject_online_call(*args, **kwargs):
        raise AssertionError("periodic feedback evaluation must not call online predict_action")

    monkeypatch.setattr(model, "forward_ttt", check_forward)
    monkeypatch.setattr(model, "predict_action", reject_online_call)
    metrics = trainer.eval_action_model({"total_loss": 1.0})
    assert metrics["total_loss"] == 1.0
    assert model.training is training
    assert model.motion_memory.training is training
    assert all(not module.training for module in model._base_modules())
    for name in ("ttt_loss", "ttt_base_loss", "ttt_future_gain", "ttt_correction_rms"):
        assert isinstance(metrics[f"train_feedback/{name}"], float)
    assert metrics["train_feedback/ttt_future_gain"] == pytest.approx(
        metrics["train_feedback/ttt_base_loss"] - metrics["train_feedback/ttt_loss"]
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_periodic_eval_restores_mode_on_failure(monkeypatch):
    from starVLA.model.modules.memory.tests.test_inverse_v3 import make_model, sample

    model = make_model().train()

    def fail(examples):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(model, "forward_ttt", fail)
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        Trainer(model, [sample()]).eval_action_model()
    assert model.training and model.motion_memory.training
    assert all(not module.training for module in model._base_modules())


@pytest.mark.parametrize("enabled", [False, None])
def test_periodic_eval_preserves_legacy_path(enabled):
    model = SimpleNamespace() if enabled is None else SimpleNamespace(ttt_enabled=enabled)
    trainer = Trainer(model, None)
    previous_metrics = {"total_loss": 2.0}
    assert trainer.eval_action_model(previous_metrics) == {"legacy": True}
    assert trainer.legacy_metrics is previous_metrics


def prepare_config(tmp_path, update):
    cfg = OmegaConf.load("examples/LIBERO_World/train_files/starvla_qwen_residual_world_inverse_v2.yaml")
    update(cfg)
    base, checkpoint, output = [tmp_path / name for name in ("v2.yaml", "model.pt", "v3.yaml")]
    OmegaConf.save(cfg, base)
    checkpoint.touch()
    prepare(base, checkpoint, output)
    return OmegaConf.load(output)


@pytest.mark.parametrize("horizon", [4, 16])
def test_config_derives_triplet_from_inherited_action_horizon(tmp_path, horizon):
    cfg = prepare_config(tmp_path, lambda cfg: setattr(cfg.framework.action_model, "action_horizon", horizon))
    assert cfg.framework.action_model.action_horizon == horizon
    assert cfg.framework.inference_horizon == horizon
    assert cfg.datasets.vla_data.trajectory_fixed_mid == horizon
    assert cfg.datasets.vla_data.trajectory_fixed_future == 2 * horizon


def test_config_supports_legacy_action_horizon(tmp_path):
    def update(cfg):
        del cfg.framework.action_model.action_horizon
        cfg.framework.action_model.future_action_window_size = 3

    cfg = prepare_config(tmp_path, update)
    assert cfg.framework.inference_horizon == 4
    assert cfg.datasets.vla_data.trajectory_fixed_future == 8


@pytest.mark.parametrize("field,value,message", [
    ("datasets.vla_data.trajectory_max_horizon", 15, "two action horizons"),
    ("framework.residual_world.absorbing_horizon", 7, "absorbing_horizon"),
    ("framework.action_model.action_horizon", 0, "positive integer"),
])
def test_config_rejects_incompatible_horizon_budget(tmp_path, field, value, message):
    with pytest.raises(ValueError, match=message):
        prepare_config(tmp_path, lambda cfg: OmegaConf.update(cfg, field, value))
