"""V3-only contiguous LIBERO anchors with aligned normalized action chunks."""
import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.trajectory_dataset import SelfForcedTrajectoryDataset


class V3AnchorDataset(SelfForcedTrajectoryDataset):
    """Keep the stage-one short target aligned with the action chunk at tails."""
    def get_step_data(self, trajectory_id, base_index):
        terminal = self._episode_length(trajectory_id) - 1
        span = int(self.fixed_future)
        if terminal < span:
            raise ValueError("episode is too short for the configured V3 anchor horizons")
        if base_index > terminal - span:
            base_index = int(torch.randint(terminal - span + 1, ()).item())
        return super().get_step_data(trajectory_id, base_index)


class TTTSequenceDataset(SelfForcedTrajectoryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ttt_stride = int(self.data_cfg.get("ttt_stride", 8))
        self.ttt_anchors = int(self.data_cfg.get("ttt_anchors", 17))
        if self.ttt_stride < 1 or self.ttt_anchors < 3:
            raise ValueError("TTT requires positive stride and at least three anchors")
        if self._action_mode != "abs":
            raise ValueError("V3 sequence action packing currently requires action_mode=abs")

    @staticmethod
    def anchor_offsets(terminal, requested, stride):
        count = min(requested, terminal // stride + 1)
        if count < 3:
            raise ValueError("episode must contain at least two complete action chunks")
        return np.arange(count, dtype=np.int64) * stride

    def get_step_data(self, trajectory_id, base_index):
        terminal = self._episode_length(trajectory_id) - 1
        offsets = self.anchor_offsets(terminal, self.ttt_anchors, self.ttt_stride)
        latest_start = terminal - int(offsets[-1])
        if not 0 <= base_index <= terminal:
            raise IndexError("anchor outside episode")
        if base_index > latest_start:
            base_index = int(torch.randint(latest_start + 1, ()).item())
        keys = self.modality_keys
        changed = [k for modality in ("video", "action", "state") for k in keys.get(modality, [])]
        previous = {k: self.delta_indices[k] for k in changed}
        try:
            for k in keys["video"]:
                self.delta_indices[k] = offsets
            for k in keys["action"]:
                self.delta_indices[k] = np.arange(int(offsets[-1]), dtype=np.int64)
            for k in keys.get("state", []):
                self.delta_indices[k] = offsets[:-1]
            data = LeRobotSingleDataset.get_step_data(self, trajectory_id, base_index)
        finally:
            self.delta_indices.update(previous)
        data[self._OFFSETS_KEY] = offsets.tolist()
        data[self._FRAME_INDICES_KEY] = (base_index + offsets).tolist()
        data[self._TERMINAL_INDEX_KEY] = terminal
        return data

    def _pack_sample(self, data):
        sample = super()._pack_sample(data)
        trajectory = sample["trajectory"]
        count = len(trajectory["images"]) - 1
        trajectory["actions"] = sample["action"].reshape(count, self.ttt_stride, -1)
        sample["action"] = trajectory["actions"][0]
        if "state" in sample:
            trajectory["states"] = sample["state"].reshape(count, 1, -1)
            sample["state"] = trajectory["states"][0]
        return sample
