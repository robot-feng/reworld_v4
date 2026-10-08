#!/usr/bin/env python3
"""Thin command entrypoint for the isolated world-evaluation plugin."""

from examples.LIBERO_World.world_eval.runner import build_parser, main


if __name__ == "__main__":
    main(build_parser().parse_args())
