#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
exec bash "${STARVLA_DIR}/examples/LIBERO/eval_files/run_policy_server.sh" "$@"
