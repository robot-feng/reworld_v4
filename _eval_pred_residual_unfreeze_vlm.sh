#!/usr/bin/env bash
# Evaluate unfreeze-VLM Pred-Residual compares ①②③ on 4 LIBERO suites.
# include_state=ON (matches training). Avoid GPU7 (occupied by other job).
# Wave1: mode1 on 0-3; mode2 on 4-6 + then libero_10 on first free of 4-6.
# Wave2: mode3 on 0-3.
set -euo pipefail
cd /9950backfile/zhangyafei/reworld_v4

STARVLA_DIR=/9950backfile/zhangyafei/reworld_v4
STARVLA_PYTHON=/9950backfile/zhangyafei/envs/qwen35vla311/bin/python
LIBERO_PYTHON=/9950backfile/zhangyafei/envs/libero/bin/python
LIBERO_HOME=/9950backfile/zhangyafei/LIBERO
ROBOSUITE_PP=/9950backfile/zhangyafei/envs/libero/lib/python3.10/site-packages
NUM_TRIALS=50
EVAL_TAG=evaluation_with_state
SUITES=(libero_spatial libero_object libero_goal libero_10)

eval_one() {
  local label="$1" gpu="$2" port="$3" suite="$4"
  local ckpt="${STARVLA_DIR}/playground/Checkpoints/${label}/final_model/pytorch_model.pt"
  local out_dir="${STARVLA_DIR}/playground/Checkpoints/${label}/${EVAL_TAG}/${suite}"
  mkdir -p "${out_dir}"
  local server_log="${out_dir}/server.log"
  local eval_log="${out_dir}/eval.log"
  rm -f "${server_log}" "${eval_log}"

  echo "[$(date '+%H:%M:%S')] START ${label} | ${suite} | GPU ${gpu} | port ${port} | include_state"

  env -u DEBUG CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${STARVLA_DIR}" \
    PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt}" --port "${port}" --use_bf16 --idle_timeout -1 \
    >"${server_log}" 2>&1 &
  local server_pid=$!

  for _ in $(seq 1 300); do
    kill -0 "${server_pid}" 2>/dev/null || { echo "  Server died for ${label}/${suite}"; tail -40 "${server_log}"; return 1; }
    ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN && break
    sleep 1
  done
  if ! ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    echo "  Server timed out for ${label}/${suite}"; kill "${server_pid}" 2>/dev/null; return 1
  fi

  env -u DEBUG PYTHONPATH="${STARVLA_DIR}" "${STARVLA_PYTHON}" \
    examples/LIBERO_World/eval_files/warm_policy_server.py \
    --port "${port}" >>"${server_log}" 2>&1 || true

  env -u DEBUG PYTHONPATH="${ROBOSUITE_PP}:${LIBERO_HOME}:${STARVLA_DIR}" \
    LIBERO_CONFIG_PATH=/home/zhangyafei/.libero \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${gpu}" \
    NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 \
    "${LIBERO_PYTHON}" examples/LIBERO_World/eval_files/eval_libero.py \
    --args.pretrained-path "${ckpt}" --args.host 127.0.0.1 --args.port "${port}" \
    --args.task-suite-name "${suite}" --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.max-tasks -1 --args.seed 7 --args.video-out-path "${out_dir}" \
    --args.no-record-video --args.include-state \
    >"${eval_log}" 2>&1
  local rc=$?

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  if [ ${rc} -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] DONE ${label} | ${suite}"
    grep "Total success rate" "${eval_log}" | tail -1
  else
    echo "[$(date '+%H:%M:%S')] FAIL ${label} | ${suite} (rc=${rc})"
    tail -40 "${eval_log}"
  fi
  echo "---"
}

run_label_on_gpus() {
  local label="$1"; shift
  local -a gpus=("$@")
  local pids=()
  local i=0
  local mode_num="${label##*_}"
  # label ends with unfreeze_vlm; extract 1/2/3 from compare_pred_residual_N_unfreeze_vlm
  mode_num=$(echo "${label}" | sed -n 's/.*residual_\([0-9]\).*/\1/p')
  for suite in "${SUITES[@]}"; do
    local gpu="${gpus[$i]}"
    local port=$((19100 + mode_num * 10 + i))
    eval_one "${label}" "${gpu}" "${port}" "${suite}" &
    pids+=($!)
    i=$((i + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || fail=1
  done
  return ${fail}
}

echo "========================================"
echo "Wave 1a: compare_pred_residual_1_unfreeze_vlm on GPUs 0-3"
echo "========================================"
run_label_on_gpus compare_pred_residual_1_unfreeze_vlm 0 1 2 3 &
pid1=$!

echo "========================================"
echo "Wave 1b: compare_pred_residual_2_unfreeze_vlm spatial/object/goal on 4-6, then long on 4"
echo "========================================"
(
  label=compare_pred_residual_2_unfreeze_vlm
  eval_one "${label}" 4 19120 libero_spatial &
  pids=( $! )
  eval_one "${label}" 5 19121 libero_object &
  pids+=( $! )
  eval_one "${label}" 6 19122 libero_goal &
  pids+=( $! )
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
  # Long after first wave of mode2 finishes (reuse GPU 4)
  eval_one "${label}" 4 19123 libero_10 || true
) &
pid2=$!

wait ${pid1} || true
wait ${pid2} || true

echo "========================================"
echo "Wave 2: compare_pred_residual_3_unfreeze_vlm on GPUs 0-3"
echo "========================================"
run_label_on_gpus compare_pred_residual_3_unfreeze_vlm 0 1 2 3 || true

echo ""
echo "========================================"
echo "Summary Pred Residual Unfreeze-VLM (with state)"
for label in compare_pred_residual_1_unfreeze_vlm compare_pred_residual_2_unfreeze_vlm compare_pred_residual_3_unfreeze_vlm; do
  echo "---- ${label} ----"
  for suite in "${SUITES[@]}"; do
    log="${STARVLA_DIR}/playground/Checkpoints/${label}/${EVAL_TAG}/${suite}/eval.log"
    sr=$(grep "Total success rate" "${log}" 2>/dev/null | tail -1 || echo "N/A")
    echo "  ${suite}: ${sr}"
  done
done
echo "========================================"
