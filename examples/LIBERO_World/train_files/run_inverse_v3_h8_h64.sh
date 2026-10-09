#!/usr/bin/env bash
# Recommended 8/64 entrypoint: one joint training run.
set -euo pipefail
exec bash "$(dirname "$0")/run_inverse_v3_joint.sh" "$@"
