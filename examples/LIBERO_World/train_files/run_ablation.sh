#!/usr/bin/env bash
# Ablation: joint vs two-stage training for QwenResidualWorldPrefill on LIBERO.
#
# Usage:
#   EXP=a GPU=6 bash run_ablation.sh          # Exp A: joint (GPU 6)
#   EXP=b1 GPU=7 bash run_ablation.sh         # Exp B stage 1: world only (GPU 7)
#   EXP=b2 GPU=7 bash run_ablation.sh         # Exp B stage 2: action (GPU 7, after b1 done)
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ENV_PY="${ENV_PY:-/9950backfile/zhangyafei/envs/qwen35vla311/bin}"
EXP="${EXP:-a}"
GPU="${GPU:-6}"
MASTER_PORT="${MASTER_PORT:-29500}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MASTER_PORT="${MASTER_PORT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

cd "${STARVLA_DIR}"

case "${EXP}" in
  a)
    CONFIG=examples/LIBERO_World/train_files/exp_a_joint.yaml
    RUN_ID=exp_a_joint_abs
    ;;
  k8)
    CONFIG=examples/LIBERO_World/train_files/exp_k8_anchor.yaml
    RUN_ID=exp_k8_anchor_abs
    ;;
  b1)
    CONFIG=examples/LIBERO_World/train_files/exp_b_stage1_world_only.yaml
    RUN_ID=exp_b_stage1_world_abs
    ;;
  b2)
    CONFIG=examples/LIBERO_World/train_files/exp_b_stage2_action.yaml
    RUN_ID=exp_b_stage2_action_abs
    STAGE1_DIR="${STARVLA_DIR}/playground/Checkpoints/exp_b_stage1_world_abs"
    CKPT=$(ls -t "${STAGE1_DIR}"/checkpoints/steps_*_pytorch_model.pt 2>/dev/null | head -1)
    if [ -z "${CKPT}" ]; then
      CKPT=$(ls -t "${STAGE1_DIR}"/final_model/pytorch_model.pt 2>/dev/null | head -1)
    fi
    if [ -z "${CKPT}" ]; then
      echo "ERROR: No stage 1 checkpoint found in ${STAGE1_DIR}"
      exit 1
    fi
    echo "Loading stage 1 checkpoint: ${CKPT}"
    ;;
  *)
    echo "Unknown EXP=${EXP}. Use a, k8, b1, or b2."
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [ "${EXP}" = "b2" ]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${CKPT}")
fi

"${ENV_PY}/accelerate" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --run_id "${RUN_ID}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
