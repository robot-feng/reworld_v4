#!/usr/bin/env bash
# Evaluate Exp A (joint) and Exp B (two-stage) on 4 LIBERO suites.
# Each experiment runs on a separate GPU, 4 suites sequentially per GPU.
#
# Usage:
#   bash run_ablation_eval.sh
#
# Override defaults:
#   GPU_A=3 GPU_B=7 NUM_TRIALS=50 bash run_ablation_eval.sh
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/9950backfile/zhangyafei/envs/qwen35vla311/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/9950backfile/zhangyafei/envs/libero/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/9950backfile/zhangyafei/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
ROBOSUITE_PYTHONPATH="${ROBOSUITE_PYTHONPATH:-$(dirname "$(dirname "$($LIBERO_PYTHON -c 'import robosuite; print(robosuite.__file__)'  2>/dev/null)")")}"

GPU_A="${GPU_A:-3}"
GPU_B="${GPU_B:-7}"
PORT_A="${PORT_A:-18300}"
PORT_B="${PORT_B:-18400}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"

CKPT_A="${STARVLA_DIR}/playground/Checkpoints/exp_a_joint_abs/final_model/pytorch_model.pt"
CKPT_B="${STARVLA_DIR}/playground/Checkpoints/exp_b_stage2_action_abs/final_model/pytorch_model.pt"

SUITES=(libero_spatial libero_object libero_goal libero_10)

eval_one() {
  local label="$1" ckpt="$2" gpu="$3" base_port="$4" suite="$5"
  local suite_idx=0
  case "${suite}" in libero_spatial) suite_idx=0;; libero_object) suite_idx=1;; libero_goal) suite_idx=2;; libero_10) suite_idx=3;; esac
  local port=$((base_port + suite_idx))
  local out_dir="${STARVLA_DIR}/playground/Checkpoints/${label}/evaluation/${suite}"
  mkdir -p "${out_dir}"

  local server_log="${out_dir}/server.log"
  local eval_log="${out_dir}/eval.log"
  rm -f "${server_log}" "${eval_log}"

  echo "[$(date '+%H:%M:%S')] ${label} | ${suite} | GPU ${gpu} | port ${port}"

  env -u DEBUG CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${STARVLA_DIR}" \
    PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt}" --port "${port}" --use_bf16 --idle_timeout -1 \
    >"${server_log}" 2>&1 &
  local server_pid=$!

  for _ in $(seq 1 180); do
    kill -0 "${server_pid}" 2>/dev/null || { echo "Server died"; tail -30 "${server_log}"; return 1; }
    ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN && break
    sleep 1
  done
  if ! ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    echo "Server timed out"; kill "${server_pid}" 2>/dev/null; return 1
  fi

  env -u DEBUG PYTHONPATH="${STARVLA_DIR}" "${STARVLA_PYTHON}" \
    examples/LIBERO_World/eval_files/warm_policy_server.py \
    --port "${port}" >>"${server_log}" 2>&1 || true

  env -u DEBUG PYTHONPATH="${ROBOSUITE_PYTHONPATH}:${LIBERO_HOME}:${STARVLA_DIR}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    MUJOCO_GL="${RENDER_BACKEND}" PYOPENGL_PLATFORM="${RENDER_BACKEND}" \
    MUJOCO_EGL_DEVICE_ID="${gpu}" NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 \
    "${LIBERO_PYTHON}" examples/LIBERO_World/eval_files/eval_libero.py \
    --args.pretrained-path "${ckpt}" --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.max-tasks -1 --args.seed 7 --args.video-out-path "${out_dir}" \
    --args.no-record-video >"${eval_log}" 2>&1
  local rc=$?

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  if [ ${rc} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ${label} | ${suite} | DONE"
    grep -i "success\|average" "${eval_log}" | tail -5
  else
    echo "[$(date '+%H:%M:%S')] ${label} | ${suite} | FAILED (rc=${rc})"
    tail -20 "${eval_log}"
  fi
  echo "---"
}

run_all_suites() {
  local label="$1" ckpt="$2" gpu="$3" base_port="$4"
  for suite in "${SUITES[@]}"; do
    eval_one "${label}" "${ckpt}" "${gpu}" "${base_port}" "${suite}"
  done
}

cd "${STARVLA_DIR}"
echo "========================================"
echo "LIBERO Ablation Evaluation"
echo "  Exp A: ${CKPT_A}"
echo "  Exp B: ${CKPT_B}"
echo "  Suites: ${SUITES[*]}"
echo "  Trials per task: ${NUM_TRIALS}"
echo "========================================"

run_all_suites "exp_a_joint_abs" "${CKPT_A}" "${GPU_A}" "${PORT_A}" &
PID_A=$!

run_all_suites "exp_b_stage2_action_abs" "${CKPT_B}" "${GPU_B}" "${PORT_B}" &
PID_B=$!

wait ${PID_A}
wait ${PID_B}

echo ""
echo "========================================"
echo "All evaluations complete. Collecting results..."
echo "========================================"

for label in exp_a_joint_abs exp_b_stage2_action_abs; do
  echo ""
  echo "=== ${label} ==="
  for suite in "${SUITES[@]}"; do
    log="${STARVLA_DIR}/playground/Checkpoints/${label}/evaluation/${suite}/eval.log"
    if [ -f "${log}" ]; then
      echo -n "  ${suite}: "
      grep -i "success\|average\|Success rate" "${log}" | tail -1
    else
      echo "  ${suite}: NO LOG"
    fi
  done
done
