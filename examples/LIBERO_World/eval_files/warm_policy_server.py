"""Warm one policy server before allocating a LIBERO EGL context."""

import argparse

import numpy as np

from examples.LIBERO_World.eval_files.model2libero_interface import ModelClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    client = ModelClient(host="127.0.0.1", port=args.port, action_ensemble=False)
    response = client.step({"image": [image, image], "lang": "warm up the policy"}, step=0)
    action = np.concatenate(tuple(response["raw_action"].values()))
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid warm-up action: {action}")


if __name__ == "__main__":
    main()
