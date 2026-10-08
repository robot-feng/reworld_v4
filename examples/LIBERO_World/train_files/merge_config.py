#!/usr/bin/env python3
"""Merge the residual-world overlay over the current official LIBERO config."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.merge(
        OmegaConf.load(args.base),
        OmegaConf.load(args.overlay),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output, resolve=True)
    print(args.output)


if __name__ == "__main__":
    main()
