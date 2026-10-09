"""Causal runner shared by offline unrolling and online observation feedback."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import torch
from torch import Tensor

from .motion_error_memory import MotionErrorMemory


@dataclass(frozen=True)
class PendingPrediction:
    target_steps: Tensor
    active: Tensor
    keys: Tensor
    origin: Tensor  # pooled detached observation
    base_delta: Tensor  # pooled detached baseline prediction, never adapted

    def detach(self):
        return PendingPrediction(*(x.detach() for x in
                                   (self.target_steps, self.active, self.keys, self.origin, self.base_delta)))


@dataclass(frozen=True)
class EpisodeState:
    fast_weights: Tensor
    pending: tuple[PendingPrediction, ...] = ()
    last_steps: Tensor | None = None
    feature_shape: tuple[int, ...] = ()

    def detach(self):
        """Call only at TBPTT boundaries; also truncate outstanding key graphs."""
        return replace(self, fast_weights=self.fast_weights.detach(),
                       pending=tuple(p.detach() for p in self.pending),
                       last_steps=None if self.last_steps is None else self.last_steps.detach())


def time_vector(value, batch: int, device, *, positive: bool = False) -> Tensor:
    result = torch.as_tensor(value, device=device)
    if result.ndim == 0:
        result = result.expand(batch)
    if result.shape != (batch,) or result.dtype == torch.bool or result.is_floating_point():
        raise ValueError("frame steps/horizons must be integers, scalar or [B]")
    if torch.any(result < (1 if positive else 0)):
        raise ValueError("frame steps must be nonnegative and horizons positive")
    return result.long().clone()


def observe_and_predict(memory: MotionErrorMemory, predictor: Callable, *,
                        semantic: Tensor, current: Tensor, horizons: list[Tensor],
                        steps: Tensor, state: EpisodeState | None = None,
                        semantic_mask: Tensor | None = None):
    """Resolve exactly due feedback, discard missed targets, then query/enqueue.

    Batch slots must keep their episode identity. Start a new episode with None;
    separate streams must carry separate states. No future targets enter this API.
    """
    b = current.shape[0]
    steps = time_vector(steps, b, current.device)
    horizons = [time_vector(h, b, current.device, positive=True) for h in horizons]
    if not horizons:
        raise ValueError("at least one prediction horizon is required")
    if any(torch.any(h == other) for i, h in enumerate(horizons) for other in horizons[:i]):
        raise ValueError("duplicate horizon queries would duplicate feedback")
    if state is None:
        state = EpisodeState(memory.init_state(b, current.device), feature_shape=tuple(current.shape))
    if state.feature_shape != tuple(current.shape) or state.fast_weights.device != current.device:
        raise ValueError("state batch/layout/device changed; reset episode state")
    if state.last_steps is not None and torch.any(steps <= state.last_steps):
        raise ValueError("observation frame steps must strictly increase in every batch slot")

    pooled = memory.pool(current.detach())
    pending, keys, values, masks = [], [], [], []
    for record in state.pending:
        due = record.active & (record.target_steps == steps)
        if due.any():
            motion = pooled - record.origin
            keys.append(record.keys)
            values.append(memory.values(motion, motion - record.base_delta))
            masks.append(due[:, None].expand(-1, record.keys.shape[1]))
        remaining = record.active & (record.target_steps > steps)
        if remaining.any():
            pending.append(replace(record, active=remaining))
    weights = state.fast_weights
    if keys:
        weights = memory.write(torch.cat(keys, 1), torch.cat(values, 1), weights, torch.cat(masks, 1))

    outputs = []
    for h in horizons:
        base = predictor(semantic, current, h, semantic_mask=semantic_mask)
        delta = base["predicted_feature_delta"]
        query = memory.address(semantic, current, delta.detach(), h, semantic_mask, query=True)
        correction = memory.read(query, weights, delta)
        adapted = delta.float() + correction
        # Preserve the actual backbone future (including its native rounding)
        # so zero correction reproduces the V2 output exactly.
        base_future = base.get("predicted_future_features")
        if base_future is None:
            base_future = current + delta
        future = base_future.float() + correction
        channel = 2 if adapted.ndim == 5 else 1
        outputs.append(dict(base, base_predicted_feature_delta=delta,
                            memory_correction=correction, predicted_feature_delta=adapted,
                            predicted_future_features=future, predicted_future_vision_features=future,
                            task_change_map=adapted.square().mean(channel).sqrt()))
        key = memory.address(semantic, current, delta.detach(), h, semantic_mask)
        pending.append(PendingPrediction(steps + h, torch.ones(b, device=steps.device, dtype=torch.bool),
                                         key, pooled, memory.pool(delta.detach())))
    return outputs, EpisodeState(weights, tuple(pending), steps, tuple(current.shape))
