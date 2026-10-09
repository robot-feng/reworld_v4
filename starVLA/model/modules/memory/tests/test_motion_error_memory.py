import pytest
import torch

from starVLA.model.modules.memory.motion_error_memory import MotionErrorMemory
from starVLA.model.modules.memory.online_residual_ttt import observe_and_predict


def make_memory():
    torch.manual_seed(17)
    return MotionErrorMemory(4, 6, dim=8, grid_size=2, gate_init=0.1)


def predict(s, z, h, semantic_mask=None):
    delta = torch.zeros_like(z) + 0.02
    return {"predicted_feature_delta": delta, "predicted_future_features": z + delta}


def step(memory, z, time, state=None, horizons=(8,)):
    return observe_and_predict(memory, predict, semantic=torch.ones(z.shape[0], 3, 6),
                               current=z, steps=time, horizons=list(horizons), state=state)


def test_gradients_through_delayed_write_and_eval_equivalence():
    memory = make_memory()
    z = torch.randn(2, 2, 4, 4, 4)
    out0, state0 = step(memory, z, 0)
    out1, state1 = step(memory, z + 0.3, 8, state0)
    out1[0]["predicted_feature_delta"].square().mean().backward()
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().max() > 0, name
    with torch.no_grad():
        _, online0 = step(memory, z, 0)
        online_out, online1 = step(memory, z + 0.3, 8, online0)
    torch.testing.assert_close(state1.fast_weights, online1.fast_weights)
    torch.testing.assert_close(out1[0]["predicted_feature_delta"], online_out[0]["predicted_feature_delta"])
    assert state0.fast_weights.count_nonzero() == 0
    torch.testing.assert_close(out0[0]["predicted_feature_delta"], predict(None, z, None)["predicted_feature_delta"], rtol=0, atol=0)
    detached = state1.detach()
    assert detached.fast_weights.grad_fn is None
    assert all(p.keys.grad_fn is None for p in detached.pending)


def test_gate_off_exact_fallback():
    memory = make_memory()
    z = torch.randn(1, 4, 4, 4)
    _, state = step(memory, z, 0)
    with torch.no_grad():
        memory.gate.zero_()
    out, _ = step(memory, z + 0.3, 8, state)
    torch.testing.assert_close(out[0]["predicted_feature_delta"], predict(None, z, None)["predicted_feature_delta"], atol=0, rtol=0)


def test_due_missed_and_batch_isolation():
    memory = make_memory()
    z = torch.randn(2, 4, 4, 4)
    _, state = step(memory, z, 0, horizons=(8, 16))
    _, early = step(memory, z, 4, state)
    assert early.fast_weights.count_nonzero() == 0
    _, mixed = step(memory, z + 0.5, [8, 9], state)
    assert mixed.fast_weights[0].count_nonzero() > 0
    assert mixed.fast_weights[1].count_nonzero() == 0  # missed frame 8 is discarded
    _, alone = step(memory, z[:1], 0, horizons=(8, 16))
    _, alone = step(memory, z[:1] + 0.5, 8, alone)
    torch.testing.assert_close(mixed.fast_weights[:1], alone.fast_weights)
    _, reset = step(memory, z + 5, 0)
    assert reset.fast_weights.count_nonzero() == 0
    assert len(reset.pending) == 1
    with pytest.raises(ValueError, match="strictly increase"):
        step(memory, z, [8, 9], mixed)
    with pytest.raises(ValueError, match="layout"):
        step(memory, z[:1], 10, mixed)


def test_feedback_averages_all_due_records_once():
    memory = make_memory()
    z = torch.randn(1, 4, 4, 4)
    _, state = step(memory, z, 0, horizons=(8, 16))
    _, state = step(memory, z + 0.1, 8, state)
    due = list(state.pending)
    pooled = memory.pool(z + 0.4)
    keys = torch.cat([p.keys for p in due], 1)
    values = torch.cat([memory.values(pooled - p.origin, pooled - p.origin - p.base_delta) for p in due], 1)
    expected = memory.write(keys, values, state.fast_weights, torch.ones(keys.shape[:2], dtype=torch.bool))
    _, next_state = step(memory, z + 0.4, 16, state)
    torch.testing.assert_close(next_state.fast_weights, expected)


def test_future_targets_only_change_loss_not_state_or_predictions():
    memory = make_memory()
    z = torch.randn(1, 4, 4, 4)
    results = []
    for future in (z + 0.4, z + 5):
        _, state = step(memory, z, 0)
        outputs, state = step(memory, z + 0.2, 8, state)
        pred = outputs[0]["predicted_future_features"]
        results.append((pred, state.fast_weights, (pred - future).square().mean()))
    for i in (0, 1):
        torch.testing.assert_close(results[0][i], results[1][i], rtol=0, atol=0)
    assert not torch.equal(results[0][2], results[1][2])


def test_long_updates_fp32_and_bfloat16_model():
    memory = make_memory().to(torch.bfloat16)
    z = torch.randn(1, 4, 4, 4).bfloat16()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, state = step(memory, z, 0, horizons=(1,))
        for i in range(1, 1001):
            _, state = step(memory, z + (i % 5) * 0.01, i, state, horizons=(1,))
        assert state.fast_weights.dtype == torch.float32
        assert torch.isfinite(state.fast_weights).all()
        assert state.fast_weights.norm() < 100
        assert len(state.pending) == 1


def test_bf16_sub_ulp_correction_survives_loss_and_gate_gradient():
    # A deterministic correction smaller than one BF16 ULP at the baseline.
    # The old BF16 addition returned R0 exactly despite a nonzero gradient.
    memory = MotionErrorMemory(4, 6, dim=8, grid_size=2, gate_init=0.01)
    z = torch.zeros(1, 4, 4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        memory.query.weight.zero_()
        memory.query.bias.zero_()
        memory.query.bias[0] = 1
        memory.decoder.weight.fill_(0.001)
    _, state = step(memory, z, 0)
    from dataclasses import replace
    state = replace(state, fast_weights=torch.eye(8)[None])
    outputs, _ = step(memory, z, 1, state)  # pending feedback is not yet due
    output = outputs[0]
    base = output["base_predicted_feature_delta"].float()
    adapted = output["predicted_feature_delta"]
    assert adapted.dtype == torch.float32
    assert output["memory_correction"].dtype == torch.float32
    assert (adapted != base).all()
    assert torch.equal(adapted.to(torch.bfloat16), base.to(torch.bfloat16))
    loss = adapted.square().mean()
    loss.backward()
    assert memory.gate.grad is not None and memory.gate.grad.abs() > 0
    with torch.no_grad():
        memory.gate.zero_()
    output, _ = step(memory, z, 1, state)
    torch.testing.assert_close(output[0]["predicted_feature_delta"], base, atol=0, rtol=0)
    torch.testing.assert_close(output[0]["predicted_future_features"],
                               predict(None, z, None)["predicted_future_features"].float(), atol=0, rtol=0)


def test_duplicate_horizons_are_rejected_per_batch_slot():
    with pytest.raises(ValueError, match="duplicate"):
        step(make_memory(), torch.zeros(2, 4, 4, 4), 0, horizons=([8, 8], [8, 16]))
