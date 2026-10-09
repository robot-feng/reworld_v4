import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.modules.memory.tests.test_inverse_v3 import make_model, sample


def make_sequence(n=17):
    example = sample()
    images = example['trajectory']['images']
    example['trajectory'] = {
        'images': [images[min(i, 2)] for i in range(n)],
        'observation_indices': [8 * i for i in range(n)],
        'frame_indices': [100 + 8 * i for i in range(n)],
        'actions': np.zeros((n-1, 8, 7), np.float32),
        'states': np.zeros((n-1, 1, 8), np.float32),
    }
    return example


class DifferentiableAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
    def forward(self, condition, actions, state):
        return (condition * self.scale - .1).square().mean()


def test_sequence_multiple_writes_long_feedback_gradients_and_causality():
    model = make_model().train()
    model.action_model = DifferentiableAction().eval()
    model.config.framework.ttt.sequence_training = True
    model.config.framework.ttt.horizons = [8, 64]
    model.config.framework.ttt.tbptt_steps = 4
    model.config.framework.ttt.action_loss_weight = .1
    recorded = []
    step = model.step_encoded
    def capture(*args, **kwargs):
        out, state = step(*args, **kwargs)
        recorded.append((state.fast_weights.detach().clone(), out[0]['predicted_feature_delta'].detach().clone()))
        return out, state
    model.step_encoded = capture
    out = model([make_sequence()])
    out['loss'].backward()
    assert len(recorded) == 16
    assert not torch.equal(recorded[1][0], recorded[8][0])
    for name, parameter in model.motion_memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    changed = make_sequence()
    changed['trajectory']['images'][-1] = sample(230)['trajectory']['images'][-1]
    model([changed])
    for original, new in zip(recorded[:16], recorded[16:]):
        for before, after in zip(original, new):
            torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_ragged_sequences_and_invalid_cadence():
    model = make_model()
    model.config.framework.ttt.sequence_training = True
    model.config.framework.ttt.horizons = [8, 64]
    model.config.framework.ttt.action_loss_weight = 0
    output = model([make_sequence(9), make_sequence(17)])
    assert torch.isfinite(output['loss'])
    bad = make_sequence()
    bad['trajectory']['frame_indices'][5] += 1
    with pytest.raises(ValueError, match='frame_indices'):
        model([bad])


def test_sequence_packing_preserves_action_state_alignment():
    from examples.LIBERO_World.train_files.sequence_dataset import TTTSequenceDataset
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    dataset = object.__new__(TTTSequenceDataset)
    dataset.ttt_stride = 8
    dataset._modality_keys = {'video': ['video.a'], 'action': ['action.a'], 'state': ['state.a'], 'language': ['lang']}
    dataset.tag = EmbodimentTag.FRANKA
    dataset.data_cfg = {'include_state': True}
    dataset._dataset_name = 'test'
    data = {
        'video.a': np.zeros((9, 8, 8, 3), np.uint8),
        'action.a': np.arange(64, dtype=np.float32)[:, None],
        'state.a': np.arange(8, dtype=np.float32)[:, None], 'lang': ['move'],
        dataset._OFFSETS_KEY: list(range(0, 65, 8)),
        dataset._FRAME_INDICES_KEY: list(range(100, 165, 8)),
        dataset._TERMINAL_INDEX_KEY: 170,
    }
    packed = dataset._pack_sample(data)
    assert packed['trajectory']['actions'].shape == (8, 8, 1)
    np.testing.assert_equal(packed['trajectory']['actions'][3, :, 0], np.arange(24, 32))
    assert packed['trajectory']['states'][3, 0, 0] == 3
    assert packed['state'].shape == (1, 1)


def test_sequence_training_and_policy_share_short_long_feedback():
    model = make_model().eval()
    model.long_inference_horizon = 64
    model.config.framework.ttt.sequence_training = True
    model.config.framework.ttt.horizons = [8, 64]
    model.config.framework.ttt.action_loss_weight = 0
    sequence = make_sequence()
    captured = []
    original = model.step_encoded
    def capture(*args, **kwargs):
        outputs, state = original(*args, **kwargs)
        captured.append(state.fast_weights.detach().clone())
        return outputs, state
    model.step_encoded = capture
    with torch.no_grad():
        model([sequence])
        offline = list(captured)
        state = None
        for i in range(16):
            obs = {'image': sequence['trajectory']['images'][i], 'lang': sequence['lang']}
            out = model.predict_action(obs, step=100+8*i, state=state)
            state = out['ttt_state']
            torch.testing.assert_close(state.fast_weights, offline[i], atol=0, rtol=0)
        assert len(state.pending) > 1  # short-only action mode still tracks long feedback
