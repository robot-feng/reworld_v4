import numpy as np
import pytest
import torch

from examples.LIBERO_World.eval_files.ttt_policy import EpisodePolicy


class Framework:
    ttt_enabled = True

    def __init__(self):
        self.seen = []

    def predict_action(self, examples, step, state, **kwargs):
        self.seen.append((step, state))
        return {"ttt_state": (step, state), "normalized_actions": np.zeros((1, 8, 7))}


def test_server_episode_and_session_isolation():
    model = Framework()
    policy = EpisodePolicy(model)
    def call(session, episode, step):
        return policy.predict_action([{}], ttt_session=session, ttt_episode=episode, step=step)
    assert "ttt_state" not in call("a", 1, 0)
    call("a", 1, 8)
    call("b", 1, 0)
    call("a", 2, 0)
    assert model.seen == [(0, None), (8, (0, None)), (0, None), (0, None)]
    with pytest.raises(ValueError, match="stale"):
        call("a", 1, 16)


def test_libero_chunk_frame_numbers_reset_and_legacy(monkeypatch):
    from examples.LIBERO.eval_files import model2libero_interface as official
    from examples.LIBERO_World.eval_files.model2libero_interface import ModelClient
    class Wire:
        enabled = True
        def __init__(self, *args):
            self.requests = []
        def get_server_metadata(self):
            return {"action_chunk_size": 8, "ttt_enabled": self.enabled}
        def predict_action(self, request):
            self.requests.append(request)
            return {"data": {"actions": np.zeros((1, 8, 7))}}
    monkeypatch.setattr(official, "WebsocketClientPolicy", Wire)
    client = ModelClient(action_ensemble=False)
    sample = {"lang": "move", "image": []}
    client.reset("move")
    for i in range(17):
        client.step(sample, step=i)
    assert [r["step"] for r in client.client.requests] == [0, 8, 16]
    assert len({r["ttt_episode"] for r in client.client.requests}) == 1
    previous_episode = client.client.requests[-1]["ttt_episode"]
    client.reset("move")
    client.step(sample, step=0)
    assert client.client.requests[-1]["ttt_episode"] == previous_episode + 1
    with pytest.raises(ValueError, match="chunk boundaries"):
        client._predict_with_state({"horizon": 12})
    Wire.enabled = False
    client = ModelClient(action_ensemble=False)
    client.step(sample, step=0)
    assert "step" not in client.client.requests[0]
    assert "ttt_session" not in client.client.requests[0]


def test_world_eval_contract():
    from starVLA.model.modules.memory.tests.test_inverse_v3 import make_model, sample
    from examples.LIBERO_World.world_eval.metrics import batch_world_metrics
    model = make_model().eval()
    example = sample()
    example["action"] = np.zeros((8, 7), dtype=np.float32)
    with torch.inference_mode():
        metrics = batch_world_metrics(model.evaluate_world([example]))
    for name in ("ttt_base_future_mse", "ttt_adapted_future_mse", "ttt_future_gain", "mid_future_mse"):
        assert metrics[name].shape == (1,)
        assert torch.isfinite(metrics[name]).all()
    torch.testing.assert_close(metrics["ttt_future_gain"], metrics["ttt_base_future_mse"] - metrics["ttt_adapted_future_mse"])


def test_fixed_triplet_sampling():
    from starVLA.dataloader.gr00t_lerobot.trajectory_dataset import SelfForcedTrajectoryDataset
    dataset = object.__new__(SelfForcedTrajectoryDataset)
    dataset.max_horizon = 500
    dataset.fixed_mid = 8
    dataset.fixed_future = 16
    assert dataset.sample_observation_indices(100) == (0, 8, 16)
    assert dataset.sample_observation_indices(12) == (0, 8, 12)
    assert dataset.sample_observation_indices(5) == (0, 4, 5)


def test_config_preserves_v2_architecture_and_data(tmp_path):
    from omegaconf import OmegaConf
    from examples.LIBERO_World.train_files.prepare_inverse_v3_config import prepare
    cfg = OmegaConf.load("examples/LIBERO_World/train_files/starvla_qwen_residual_world_inverse_v2.yaml")
    cfg.framework.residual_world.hidden_dim = 256
    cfg.framework.vision_view_indices = [1]
    cfg.datasets.vla_data.data_root_dir = "/test/libero"
    base, checkpoint, output = (tmp_path / name for name in ("v2.yaml", "model.pt", "v3.yaml"))
    OmegaConf.save(cfg, base)
    checkpoint.touch()
    prepare(base, checkpoint, output)
    v3 = OmegaConf.load(output)
    assert v3.framework.name == "QwenResidualWorldInverseV3"
    assert v3.framework.residual_world == cfg.framework.residual_world
    assert v3.framework.vision_view_indices == [1]
    assert v3.datasets.vla_data.data_root_dir == "/test/libero"
    assert v3.datasets.vla_data.trajectory_fixed_future == 16
    assert v3.trainer.learning_rate.motion_memory == 1e-4
    assert v3.trainer.pretrained_checkpoint == str(checkpoint)
