#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: $0 CHECKPOINT TASK_SUITE GPU PORT OUTPUT_DIR"
  exit 2
fi

CKPT="$1"
TASK_SUITE="$2"
GPU="$3"
PORT="$4"
OUTPUT_DIR="$5"
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/starVLA/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${STARVLA_PYTHON}}"
LIBERO_HOME="${LIBERO_HOME:-$(cd "${STARVLA_DIR}/../../.." && pwd)/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
ROBOSUITE_PYTHONPATH="${ROBOSUITE_PYTHONPATH:-}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
SEED="${SEED:-7}"
RECORD_VIDEO="${RECORD_VIDEO:-false}"
WORLD_HORIZON="${WORLD_HORIZON:-}"
START_BARRIER="${START_BARRIER:-}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"

if [[ -z "${ROBOSUITE_PYTHONPATH}" ]]; then
  SINGLE_ARM_ENV="$(find "${HOME}/.cache/uv" -path '*/robosuite/environments/manipulation/single_arm_env.py' -print -quit 2>/dev/null || true)"
  [[ -n "${SINGLE_ARM_ENV}" ]] && ROBOSUITE_PYTHONPATH="$(dirname "$(dirname "$(dirname "$(dirname "${SINGLE_ARM_ENV}")")")")"
fi
if [[ ! -f "${CKPT}" || ! -f "${LIBERO_CONFIG_PATH}/config.yaml" || -z "${ROBOSUITE_PYTHONPATH}" ]]; then
  echo "Missing checkpoint, LIBERO config, or robosuite 1.4.x path"
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/server.log"
EVAL_LOG="${OUTPUT_DIR}/eval.log"
rm -f "${SERVER_LOG}" "${EVAL_LOG}"
server_pid=""
cleanup() {
  [[ -z "${server_pid}" ]] || kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env -u DEBUG CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${STARVLA_DIR}" \
  PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${STARVLA_PYTHON}" examples/LIBERO_World/eval_files/server_policy.py --ckpt_path "${CKPT}" \
  --port "${PORT}" --use_bf16 --idle_timeout -1 >"${SERVER_LOG}" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  kill -0 "${server_pid}" 2>/dev/null || { tail -n 80 "${SERVER_LOG}"; exit 1; }
  ss -ltn "sport = :${PORT}" | rg -q LISTEN && break
  sleep 1
done
ss -ltn "sport = :${PORT}" | rg -q LISTEN || { echo "Policy server timed out on port ${PORT}"; exit 1; }
env -u DEBUG PYTHONPATH="${STARVLA_DIR}" "${STARVLA_PYTHON}" \
  examples/LIBERO_World/eval_files/warm_policy_server.py --port "${PORT}" >>"${SERVER_LOG}" 2>&1

if [[ -n "${START_BARRIER}" ]]; then
  mkdir -p "${START_BARRIER}"
  touch "${START_BARRIER}/ready_${PORT}"
  for _ in $(seq 1 600); do
    [[ -f "${START_BARRIER}/start" ]] && break
    kill -0 "${server_pid}" 2>/dev/null || exit 1
    sleep 1
  done
  [[ -f "${START_BARRIER}/start" ]] || { echo "Evaluation start barrier timed out"; exit 1; }
fi

video_arg="--args.no-record-video"
[[ "${RECORD_VIDEO,,}" =~ ^(1|true|yes)$ ]] && video_arg="--args.record-video"
world_horizon_arg=()
[[ -z "${WORLD_HORIZON}" ]] || world_horizon_arg=(--args.world-horizon "${WORLD_HORIZON}")
env -u DEBUG PYTHONPATH="${ROBOSUITE_PYTHONPATH}:${LIBERO_HOME}:${STARVLA_DIR}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" MUJOCO_GL="${RENDER_BACKEND}" PYOPENGL_PLATFORM="${RENDER_BACKEND}" \
  MUJOCO_EGL_DEVICE_ID="${GPU}" NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 \
  "${LIBERO_PYTHON}" examples/LIBERO_World/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" --args.host 127.0.0.1 --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" --args.seed "${SEED}" --args.video-out-path "${OUTPUT_DIR}" \
  "${video_arg}" "${world_horizon_arg[@]}" >"${EVAL_LOG}" 2>&1
