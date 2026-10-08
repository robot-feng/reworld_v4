#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 WORLD_LOSS_0_CHECKPOINT WORLD_LOSS_025_CHECKPOINT"
  exit 2
fi

CKPT_WORLD000="$(realpath "$1")"
CKPT_WORLD025="$(realpath "$2")"
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKER="${STARVLA_DIR}/examples/LIBERO_World/eval_files/run_eval_job.sh"
BASE_PORT="${BASE_PORT:-18200}"
RUN_NAME="${RUN_NAME:-task_success_full_50_no_video}"
BARRIER_DIR="$(mktemp -d "/tmp/starvla_libero_${BASE_PORT}.XXXXXX")"
EIGHT_WAY_RENDER_BACKEND="${EIGHT_WAY_RENDER_BACKEND:-osmesa}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
GPUS=(0 1 2 3)
RENDER_BACKENDS=("${EIGHT_WAY_RENDER_BACKEND}" "${EIGHT_WAY_RENDER_BACKEND}" "${EIGHT_WAY_RENDER_BACKEND}" "${EIGHT_WAY_RENDER_BACKEND}")

launch_group() {
  local label="$1"
  local checkpoint="$2"
  local port_offset="$3"
  local model_root="${checkpoint%%/checkpoints/*}"
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint}")"
  local step_dir="${checkpoint_name%_pytorch_model.pt}"

  for index in "${!SUITES[@]}"; do
    local suite="${SUITES[$index]}"
    local gpu="${GPUS[$index]}"
    local render_backend="${RENDER_BACKENDS[$index]}"
    local port=$((BASE_PORT + index * 10 + port_offset))
    local session="libero_${label}_${suite}_${checkpoint_name%%.*}"
    local output_dir="${model_root}/evaluation/${step_dir}/${RUN_NAME}/results/${suite}/${checkpoint_name}"
    tmux has-session -t "${session}" 2>/dev/null && { echo "Session already exists: ${session}"; exit 1; }

    local command=(env STARVLA_DIR="${STARVLA_DIR}" STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
      LIBERO_PYTHON="${LIBERO_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
      LIBERO_HOME="${LIBERO_HOME:-/home/taizun/tzq/LIBERO}" LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
      ROBOSUITE_PYTHONPATH="${ROBOSUITE_PYTHONPATH:-}" NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
      MAX_TASKS="${MAX_TASKS:--1}" RECORD_VIDEO="${RECORD_VIDEO:-false}" RENDER_BACKEND="${render_backend}"
      START_BARRIER="${BARRIER_DIR}" bash "${WORKER}"
      "${checkpoint}" "${suite}" "${gpu}" "${port}" "${output_dir}")
    printf -v shell_command '%q ' "${command[@]}"
    tmux new-session -d -s "${session}" -c "${STARVLA_DIR}" "${shell_command}"
    echo "${session}: GPU ${gpu}, port ${port}, ${suite}, ${label}, ${render_backend}"
  done
}

wait_for_ready() {
  local expected="$1"
  for _ in $(seq 1 300); do
    local ready
    ready="$(find "${BARRIER_DIR}" -maxdepth 1 -name 'ready_*' | wc -l)"
    (( ready >= expected )) && return
    sleep 1
  done
  echo "Timed out waiting for ${expected} warm policy servers"
  exit 1
}

launch_group world000 "${CKPT_WORLD000}" 0
wait_for_ready 4
launch_group world025 "${CKPT_WORLD025}" 1
wait_for_ready 8
touch "${BARRIER_DIR}/start"

echo "Eight warmed policy servers released to LIBERO. Use: tmux ls | rg '^libero_world'"
