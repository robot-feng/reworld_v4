"""V3 sequential feedback losses; all writes use already observed frames only."""
from collections import defaultdict

import torch
from torch.nn import functional as F


def forward_sequence(model, examples):
    groups = defaultdict(list)
    cfg = model.config.framework.ttt
    horizons = [int(h) for h in cfg.get("horizons", [model.action_horizon, model.long_inference_horizon])]
    stride = model.action_horizon
    if len(horizons) != len(set(horizons)) or any(h < 1 or h % stride for h in horizons):
        raise ValueError("sequence horizons must be unique positive multiples of action_horizon")
    if stride not in horizons:
        raise ValueError("sequence training must include action_horizon")
    tbptt = int(cfg.get("tbptt_steps", 4))
    action_weight = float(cfg.get("action_loss_weight", 0.1))
    if tbptt < 1 or action_weight < 0:
        raise ValueError("invalid TBPTT or action loss weight")
    for sample in examples:
        trajectory = sample["trajectory"]
        offsets = trajectory["observation_indices"]
        frames = trajectory.get("frame_indices", offsets)
        n = len(offsets)
        if n < 3 or len(trajectory["images"]) != n or len(frames) != n:
            raise ValueError("sequence requires matching images/indices with at least 3 anchors")
        if offsets != list(range(0, n * stride, stride)):
            raise ValueError("sequence observation indices must follow action_horizon cadence")
        if any(f - frames[0] != o for f, o in zip(frames, offsets)):
            raise ValueError("frame_indices must agree with observation_indices")
        if action_weight and ("actions" not in trajectory or len(trajectory["actions"]) != n - 1):
            raise ValueError("sequence action loss requires aligned action chunks at every anchor")
        groups[n].append(sample)

    totals = None
    for n, group in groups.items():
        state, supervision = None, []
        losses, base_losses, action_losses, corrections = [], [], [], []
        for index in range(n):
            observations = []
            for sample in group:
                trajectory = sample["trajectory"]
                observation = {"image": trajectory["images"][index], "lang": sample["lang"]}
                if index < n - 1 and "states" in trajectory:
                    observation["state"] = trajectory["states"][index]
                observations.append(observation)
            if index == n - 1:
                batch = model._prepare_batch(observations, training=False)
                with torch.no_grad():
                    current = model._encode_vision(batch.world_images)
            else:
                batch, semantic, mask, current = model._encode_observation(observations)
            # Score pending outputs before the current frame can update memory.
            remaining = []
            for target_index, origin, output in supervision:
                if target_index == index:
                    target = current.float() - origin.float()
                    losses.append(F.mse_loss(output["predicted_feature_delta"], target))
                    base_losses.append(F.mse_loss(output["base_predicted_feature_delta"].float(), target))
                else:
                    remaining.append((target_index, origin, output))
            supervision = remaining
            if index == n - 1:
                break
            frames = [s["trajectory"].get("frame_indices", s["trajectory"]["observation_indices"])[index] for s in group]
            outputs, state = model.step_encoded(semantic, current, steps=frames, horizons=horizons,
                                                state=state, semantic_mask=mask)
            if index > 0:
                by_h = dict(zip(horizons, outputs))
                for h, output in by_h.items():
                    target_index = index + h // stride
                    if target_index < n:
                        supervision.append((target_index, current.detach(), output))
                    corrections.append(output["memory_correction"].float().square().mean())
                if action_weight:
                    short = by_h[stride]
                    long = by_h.get(model.long_inference_horizon)
                    condition = model._build_action_condition(
                        current=current, short_delta=short["predicted_feature_delta"].to(current.dtype),
                        long_delta=None if long is None else long["predicted_feature_delta"].to(current.dtype),
                        short_features=short["predicted_future_features"].to(current.dtype),
                    )
                    actions = model._tensor_field([{"action": s["trajectory"]["actions"][index]} for s in group], "action")
                    # Frozen parameters still allow gradients to the memory input.
                    action_losses.append(model.action_model(condition, model._action_targets(actions), batch.state))
            if (index + 1) % tbptt == 0:
                state = state.detach()
        world_loss = torch.stack(losses).mean()
        base_loss = torch.stack(base_losses).mean().detach()
        action_loss = torch.stack(action_losses).mean() if action_losses else world_loss.new_zeros(())
        values = {
            "loss": world_loss + action_weight * action_loss,
            "ttt_loss": world_loss,
            "ttt_base_loss": base_loss,
            "ttt_action_loss": action_loss,
            "ttt_future_gain": (base_loss - world_loss).detach(),
            "ttt_correction_rms": torch.stack(corrections).mean().sqrt().detach(),
        }
        weight = len(group) / len(examples)
        totals = {k: v * weight for k, v in values.items()} if totals is None else {
            k: totals[k] + v * weight for k, v in values.items()}
    return totals
