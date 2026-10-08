#!/usr/bin/env bash
# Launch ①②③ Pred-Residual compares in parallel (2 GPUs each).
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"
SCRIPT="${STARVLA_DIR}/examples/LIBERO_World/train_files/run_pred_residual_compare.sh"
LOG_DIR="${STARVLA_DIR}/playground/Checkpoints"
mkdir -p "${LOG_DIR}"

launch_one() {
  local mode="$1" gpus="$2"
  local log="${LOG_DIR}/_compare_pred_residual_${mode}.log"
  echo "[$(date '+%H:%M:%S')] launching MODE=${mode} on GPUs ${gpus} -> ${log}"
  nohup env MODE="${mode}" GPUS="${gpus}" bash "${SCRIPT}" >"${log}" 2>&1 &
  echo "  pid=$!"
}

launch_one 1 0,1
launch_one 2 2,3
launch_one 3 4,5

echo "All three launched. Tail logs with:"
echo "  tail -f ${LOG_DIR}/_compare_pred_residual_{1,2,3}.log"
