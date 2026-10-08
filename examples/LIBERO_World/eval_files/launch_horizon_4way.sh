#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: WORLD_HORIZON=N $0 CHECKPOINT"
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
WORLD_HORIZON="${WORLD_HORIZON:?Set WORLD_HORIZON to a positive integer}"
[[ "${WORLD_HORIZON}" =~ ^[1-9][0-9]*$ ]] || { echo "WORLD_HORIZON must be a positive integer"; exit 2; }

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKER="${STARVLA_DIR}/examples/LIBERO_World/eval_files/run_eval_job.sh"
BASE_PORT="${BASE_PORT:-18600}"
LABEL="${LABEL:-world025}"
RUN_NAME="${RUN_NAME:-task_success_horizon_${WORLD_HORIZON}_full_50_no_video}"
BARRIER_DIR="$(mktemp -d "/tmp/starvla_libero_h${WORLD_HORIZON}_${BASE_PORT}.XXXXXX")"
SUITES=(libero_spatial libero_object libero_goal libero_10)
IFS=',' read -r -a GPUS <<< "${GPU_MAP:-0,1,2,3}"
(( ${#GPUS[@]} == ${#SUITES[@]} )) || { echo "GPU_MAP must contain four comma-separated GPU indices"; exit 2; }

model_root="${CHECKPOINT%%/checkpoints/*}"
checkpoint_name="$(basename "${CHECKPOINT}")"
step_dir="${checkpoint_name%_pytorch_model.pt}"

for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  gpu="${GPUS[$index]}"
  port=$((BASE_PORT + index * 10))
  session="libero_${LABEL}_${suite}_${checkpoint_name%%.*}_h${WORLD_HORIZON}"
  output_dir="${model_root}/evaluation/${step_dir}/${RUN_NAME}/results/${suite}/${checkpoint_name}"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Session already exists: ${session}"; exit 1; }

  command=(env STARVLA_DIR="${STARVLA_DIR}" STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
    LIBERO_PYTHON="${LIBERO_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
    LIBERO_HOME="${LIBERO_HOME:-/home/taizun/tzq/LIBERO}" LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
    ROBOSUITE_PYTHONPATH="${ROBOSUITE_PYTHONPATH:-}" NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
    MAX_TASKS="${MAX_TASKS:--1}" RECORD_VIDEO="${RECORD_VIDEO:-false}" RENDER_BACKEND="${RENDER_BACKEND:-osmesa}"
    WORLD_HORIZON="${WORLD_HORIZON}" START_BARRIER="${BARRIER_DIR}" bash "${WORKER}"
    "${CHECKPOINT}" "${suite}" "${gpu}" "${port}" "${output_dir}")
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-session -d -s "${session}" -c "${STARVLA_DIR}" "${shell_command}"
  echo "${session}: GPU ${gpu}, port ${port}, horizon ${WORLD_HORIZON}"
done

for _ in $(seq 1 300); do
  ready="$(find "${BARRIER_DIR}" -maxdepth 1 -name 'ready_*' | wc -l)"
  (( ready >= 4 )) && break
  sleep 1
done
(( ready >= 4 )) || { echo "Timed out waiting for four policy servers"; exit 1; }
touch "${BARRIER_DIR}/start"
echo "Four horizon=${WORLD_HORIZON} evaluations released to LIBERO"
