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


ROBOT_TYPE_CONFIG_MAP = {
    "libero_residual_world_franka": LiberoResidualWorldDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = {}

DATASET_NAMED_MIXTURES = {
    "libero_residual_world_all": [
        (dataset_name, weight, "libero_residual_world_franka")
        for dataset_name, weight, _ in OFFICIAL_LIBERO_MIXTURES["libero_all"]
    ],
}
