"""Residual-world data registration as a thin extension of official LIBERO."""

from typing import Any

from examples.LIBERO.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES as OFFICIAL_LIBERO_MIXTURES,
    Libero4in1DataConfig,
)
from starVLA.dataloader.gr00t_lerobot.trajectory_dataset import (
    SelfForcedTrajectoryDataset,
)


class LiberoResidualWorldDataConfig(Libero4in1DataConfig):
    """Official LIBERO modalities plus opt-in ``[0, m, n]`` video sampling."""

    def make_dataset(self, **kwargs: Any):
        return SelfForcedTrajectoryDataset(**kwargs)


class LiberoTTTSequenceDataConfig(Libero4in1DataConfig):
    def make_dataset(self, **kwargs: Any):
        from examples.LIBERO_World.train_files.sequence_dataset import TTTSequenceDataset
        return TTTSequenceDataset(**kwargs)


class LiberoV3AnchorDataConfig(Libero4in1DataConfig):
    def make_dataset(self, **kwargs: Any):
        from examples.LIBERO_World.train_files.sequence_dataset import V3AnchorDataset
        return V3AnchorDataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {
    "libero_residual_world_franka": LiberoResidualWorldDataConfig(),
    "libero_ttt_sequence_franka": LiberoTTTSequenceDataConfig(),
    "libero_v3_anchor_franka": LiberoV3AnchorDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = {}

DATASET_NAMED_MIXTURES = {
    "libero_v3_anchor_all": [
        (dataset_name, weight, "libero_v3_anchor_franka")
        for dataset_name, weight, _ in OFFICIAL_LIBERO_MIXTURES["libero_all"]
    ],
    "libero_ttt_sequence_all": [
        (dataset_name, weight, "libero_ttt_sequence_franka")
        for dataset_name, weight, _ in OFFICIAL_LIBERO_MIXTURES["libero_all"]
    ],
    "libero_residual_world_all": [
        (dataset_name, weight, "libero_residual_world_franka")
        for dataset_name, weight, _ in OFFICIAL_LIBERO_MIXTURES["libero_all"]
    ],
}
