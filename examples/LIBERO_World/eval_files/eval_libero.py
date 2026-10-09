"""LIBERO-World entrypoint with optional replay-video encoding."""

import dataclasses
import logging
from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

import torch
import tyro

from examples.LIBERO.eval_files import eval_libero as official_eval
from examples.LIBERO_World.eval_files.model2libero_interface import ModelClient


@dataclasses.dataclass
class Args(official_eval.Args):
    """Official arguments plus an opt-in replay recorder."""

    record_video: bool = False
    world_horizon: int | None = None


class _WorldHorizonModelClient(ModelClient):
    """Attach a world horizon to requests without changing the official client."""

    def __init__(self, *args, world_horizon: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._world_horizon = world_horizon
        self._predict_action = self.client.predict_action
        self.client.predict_action = self._predict_with_horizon

    def _predict_with_horizon(self, request: dict) -> dict:
        return self._predict_action({**request, "horizon": self._world_horizon})


def eval_libero(args: Args) -> None:
    if args.world_horizon is not None and args.world_horizon < 1:
        raise ValueError("world_horizon must be positive")
    original_torch_load = torch.load
    original_get_env = official_eval._get_libero_env
    live_env = None

    def load_trusted_libero_state(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    def replace_env(*env_args, **env_kwargs):
        nonlocal live_env
        if live_env is not None:
            live_env.close()
        live_env, description = original_get_env(*env_args, **env_kwargs)
        return live_env, description

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch, "load", load_trusted_libero_state))
            stack.enter_context(patch.object(official_eval, "_get_libero_env", replace_env))
            if args.world_horizon is not None:
                client = partial(_WorldHorizonModelClient, world_horizon=args.world_horizon)
                stack.enter_context(patch.object(official_eval, "ModelClient", client))
            else:
                stack.enter_context(patch.object(official_eval, "ModelClient", ModelClient))
            if not args.record_video:
                logging.info("Replay-video encoding is disabled")
                stack.enter_context(patch.object(official_eval.imageio, "mimwrite", return_value=None))
            official_eval.eval_libero(args)
    finally:
        if live_env is not None:
            live_env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
