"""Opt-in ``[current, middle, target]`` sampling for world training.

Default policy pins action-chunk anchor ``K=8`` and draws a second full-range
horizon ``H ~ Beta``, then returns time-ordered
``[0, min(K, H), max(K, H)]``. Official StarVLA datasets and their default
``observation_indices=[0]`` stay untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


DEFAULT_MAX_HORIZON = 500
DEFAULT_BETA_ALPHA = 2.5
DEFAULT_BETA_BETA = 1.0
DEFAULT_ACTION_ANCHOR = 8


def sample_self_forced_indices(
    remaining_horizon: int,
    *,
    alpha: float = DEFAULT_BETA_ALPHA,
    beta: float = DEFAULT_BETA_BETA,
    action_anchor: int = DEFAULT_ACTION_ANCHOR,
) -> tuple[int, int, int]:
    """Sample ``[0, min(K, H), max(K, H)]`` with ``H != K``.

    ``H`` is drawn from a Beta-discretized support on ``{1, ..., R}``. Ties with
    the fixed action-chunk anchor ``K`` are resampled so the self-forced gap is
    never zero. When ``R < K`` (short episode tail), fall back to two distinct
    ordered Beta draws on ``{1, ..., R}``.
    """

    if remaining_horizon < 2:
        raise ValueError("remaining_horizon must contain two future frames")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta distribution parameters must be positive")
    if action_anchor < 1:
        raise ValueError("action_anchor must be positive")

    remaining = int(remaining_horizon)
    if remaining == 2:
        return 0, 1, 2

    distribution = torch.distributions.Beta(alpha, beta)

    # Short tails cannot host the fixed K=8 anchor; keep two ordered futures.
    if remaining < action_anchor:
        for _ in range(1024):
            horizons = (
                distribution.sample((2,)) * remaining
            ).ceil().clamp_(1, remaining).long()
            if int(horizons[0]) != int(horizons[1]):
                middle, target = (int(v) for v in horizons.sort().values.tolist())
                return 0, middle, target
        raise RuntimeError("Beta samples repeatedly mapped to the same discrete horizon")

    for _ in range(1024):
        # Map u~Beta onto {1, ..., R}; reject H == K so path length |H-K| >= 1.
        horizon = int((distribution.sample() * remaining).ceil().clamp_(1, remaining).item())
        if horizon == action_anchor:
            continue
        mid = min(action_anchor, horizon)
        future = max(action_anchor, horizon)
        return 0, mid, future
    raise RuntimeError("failed to sample H distinct from action_anchor")


def resolve_frame_indices(
    current_index: int,
    terminal_index: int,
    observation_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve strict relative offsets to absolute episode indices."""

    if not 0 <= current_index <= terminal_index:
        raise IndexError("current_index must be inside the episode")
    if not observation_indices or observation_indices[0] != 0:
        raise ValueError("observation_indices must start at zero")
    if any(index < 0 for index in observation_indices):
        raise ValueError("observation_indices must be non-negative")
    if any(
        left >= right
        for left, right in zip(
            observation_indices,
            observation_indices[1:],
        )
    ):
        raise ValueError("observation_indices must be strictly increasing")
    frame_indices = tuple(
        current_index + index for index in observation_indices
    )
    if frame_indices[-1] > terminal_index:
        raise IndexError("observation target crosses the episode boundary")
    return frame_indices


class SelfForcedTrajectoryDataset(LeRobotSingleDataset):
    """Return raw time-major trajectory images under ``sample['trajectory']``.

    Override :meth:`sample_observation_indices` in another opt-in subclass to
    request a different set or segment of frames. The consuming custom
    framework decides how to supervise that trajectory; official loaders and
    top-level sample fields remain unchanged.
    """

    _OFFSETS_KEY = "__trajectory_observation_indices__"
    _FRAME_INDICES_KEY = "__trajectory_frame_indices__"
    _TERMINAL_INDEX_KEY = "__trajectory_terminal_index__"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_horizon = int(
            self.data_cfg.get(
                "trajectory_max_horizon",
                self.data_cfg.get(
                    "future_target_max_horizon",
                    DEFAULT_MAX_HORIZON,
                ),
            )
        )
        self.beta_alpha = float(
            self.data_cfg.get("trajectory_beta_alpha", DEFAULT_BETA_ALPHA)
        )
        self.beta_beta = float(
            self.data_cfg.get("trajectory_beta_beta", DEFAULT_BETA_BETA)
        )
        self.action_anchor = int(
            self.data_cfg.get("trajectory_action_anchor", DEFAULT_ACTION_ANCHOR)
        )
        fixed_mid = self.data_cfg.get("trajectory_fixed_mid", None)
        self.fixed_mid = None if fixed_mid in (None, "", "null") else int(fixed_mid)
        fixed_future = self.data_cfg.get("trajectory_fixed_future", None)
        self.fixed_future = None if fixed_future in (None, "", "null") else int(fixed_future)
        if self.fixed_future is not None and (
            self.fixed_mid is None or self.fixed_future <= self.fixed_mid
            or self.fixed_future > self.max_horizon
        ):
            raise ValueError("fixed future requires fixed_mid < fixed_future <= max_horizon")
        if self.max_horizon < 2:
            raise ValueError("trajectory_max_horizon must be at least two")
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("trajectory Beta parameters must be positive")
        if self.action_anchor < 1:
            raise ValueError("trajectory_action_anchor must be positive")
        if self.fixed_mid is not None and self.fixed_mid < 1:
            raise ValueError("trajectory_fixed_mid must be a positive integer")
        for key in self.modality_keys["video"]:
            if not np.array_equal(self.delta_indices[key], np.asarray([0])):
                raise ValueError(
                    f"{key} must use observation_indices=[0]; "
                    "future offsets are sampled per item"
                )

    def sample_observation_indices(
        self,
        remaining_horizon: int,
    ) -> tuple[int, int, int]:
        """Policy-owned sampling point for this trajectory algorithm."""

        remaining = min(int(remaining_horizon), self.max_horizon)
        if self.fixed_mid is None:
            return sample_self_forced_indices(
                remaining,
                alpha=self.beta_alpha,
                beta=self.beta_beta,
                action_anchor=self.action_anchor,
            )
        # Pin short/action anchor (e.g. h=8) while keeping a longer random target.
        if remaining < 2:
            raise ValueError("fixed mid requires remaining_horizon >= 2")
        mid = min(self.fixed_mid, remaining - 1)
        if mid < 1:
            raise ValueError("trajectory_fixed_mid must leave room for a future frame")
        if self.fixed_future is not None:
            # Short episode tails retain strict ordering, as in fixed-mid mode.
            return 0, mid, min(self.fixed_future, remaining)
        if remaining == mid + 1:
            return 0, mid, mid + 1
        fraction = torch.distributions.Beta(self.beta_alpha, self.beta_beta).sample()
        span = remaining - mid
        target = mid + 1 + min(int(fraction * span), span - 1)
        return 0, mid, target

    def _episode_length(self, trajectory_id: int) -> int:
        return int(
            self.trajectory_lengths[self.get_trajectory_index(trajectory_id)]
        )

    @staticmethod
    def _valid_anchor(base_index: int, terminal_index: int) -> int:
        """Keep official sample counts while guaranteeing two future frames."""

        if not 0 <= base_index <= terminal_index:
            raise IndexError("base_index must be inside the episode")
        if terminal_index < 2:
            raise IndexError("episode must contain at least three frames")
        if base_index > terminal_index - 2:
            return int(torch.randint(0, terminal_index - 1, ()).item())
        return int(base_index)

    def get_step_data(
        self,
        trajectory_id: int,
        base_index: int,
    ) -> dict[str, Any]:
        episode_length = self._episode_length(trajectory_id)
        terminal_index = episode_length - 1
        base_index = self._valid_anchor(int(base_index), terminal_index)
        remaining_horizon = terminal_index - base_index
        offsets = self.sample_observation_indices(remaining_horizon)
        frame_indices = resolve_frame_indices(
            base_index,
            terminal_index,
            offsets,
        )

        original_offsets = {
            key: self.delta_indices[key]
            for key in self.modality_keys["video"]
        }
        try:
            for key in original_offsets:
                self.delta_indices[key] = np.asarray(offsets, dtype=np.int64)
            data = super().get_step_data(trajectory_id, base_index)
        finally:
            self.delta_indices.update(original_offsets)

        data[self._OFFSETS_KEY] = offsets
        data[self._FRAME_INDICES_KEY] = frame_indices
        data[self._TERMINAL_INDEX_KEY] = terminal_index
        return data

    @staticmethod
    def _pil(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB").resize((224, 224))
        return Image.fromarray(np.asarray(frame)).convert("RGB").resize((224, 224))

    def _pack_sample(self, data: dict[str, Any]) -> dict[str, Any]:
        sample = super()._pack_sample(data)
        offsets = tuple(int(value) for value in data[self._OFFSETS_KEY])
        frame_count = len(offsets)
        if any(len(data[key]) != frame_count for key in self.modality_keys["video"]):
            raise ValueError("every video view must contain all trajectory frames")

        images = [sample["image"]]
        images.extend(
            [self._pil(data[key][time_index]) for key in self.modality_keys["video"]]
            for time_index in range(1, frame_count)
        )
        sample["trajectory"] = {
            "images": images,
            "observation_indices": list(offsets),
            "frame_indices": [
                int(value) for value in data[self._FRAME_INDICES_KEY]
            ],
            "terminal_index": int(data[self._TERMINAL_INDEX_KEY]),
            "dataset_name": self.dataset_name,
        }
        return sample


def _smoke_test() -> None:
    torch.manual_seed(0)
    samples = [sample_self_forced_indices(100) for _ in range(2048)]
    assert all(0 < middle < target <= 100 for _, middle, target in samples)
    assert all(8 in (middle, target) for _, middle, target in samples)
    assert all(middle != target for _, middle, target in samples)
    short = [sample_self_forced_indices(5) for _ in range(256)]
    assert all(0 < middle < target <= 5 for _, middle, target in short)
    assert sample_self_forced_indices(2) == (0, 1, 2)
    assert resolve_frame_indices(8, 20, (0, 3, 10)) == (8, 11, 18)
    print("SelfForcedTrajectoryDataset K=8 smoke test passed")


if __name__ == "__main__":
    _smoke_test()
