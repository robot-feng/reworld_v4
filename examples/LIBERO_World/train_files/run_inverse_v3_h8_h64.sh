#!/usr/bin/env bash
# Recommended 8/64 entrypoint: effective per-GPU batch 32 with 8x4 accumulation.
set -euo pipefail
exec bash "$(dirname "$0")/run_inverse_v3_joint.sh" "$@"
