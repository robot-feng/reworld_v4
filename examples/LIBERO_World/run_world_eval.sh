#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:?Set CKPT to a checkpoint file or run directory}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-100}"
MAX_VISUALIZATIONS="${MAX_VISUALIZATIONS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

ARGS=(
  --checkpoint "${CKPT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --max-batches "${MAX_BATCHES}"
  --max-visualizations "${MAX_VISUALIZATIONS}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${RANDOM_CORE:-0}" == "1" ]]; then
  ARGS+=(--random-core)
fi

"${STARVLA_PYTHON}" examples/LIBERO_World/eval_world.py "${ARGS[@]}"
