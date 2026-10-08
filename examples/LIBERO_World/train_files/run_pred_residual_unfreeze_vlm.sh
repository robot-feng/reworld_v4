#!/usr/bin/env bash
# Train Pred-Residual compare with VLM unfrozen (RADIO still frozen).
#
# Usage:
#   MODE=1 GPUS=0,1 bash run_pred_residual_unfreeze_vlm.sh
#   MODE=2 GPUS=2,3 bash run_pred_residual_unfreeze_vlm.sh
#   MODE=3 GPUS=4,5 bash run_pred_residual_unfreeze_vlm.sh
set -euo pipefail

MODE="${MODE:?set MODE=1|2|3}"
case "${MODE}" in
  1|2|3) ;;
  *) echo "MODE must be 1, 2, or 3"; exit 1 ;;
esac

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ENV_PY="${ENV_PY:-/9950backfile/zhangyafei/envs/qwen35vla311/bin}"
GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-$((29700 + MODE))}"

CONFIG="${STARVLA_DIR}/examples/LIBERO_World/train_files/compare_pred_residual_${MODE}_unfreeze_vlm.yaml"
RUN_ID="${RUN_ID:-compare_pred_residual_${MODE}_unfreeze_vlm}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export MASTER_PORT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens17f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens17f0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cd "${STARVLA_DIR}"
mkdir -p "${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}"

echo "========================================"
echo "Pred Residual Unfreeze-VLM MODE=${MODE}"
echo "  config: ${CONFIG}"
echo "  run_id: ${RUN_ID}"
echo "  freeze: vision_encoder only (VLM trainable)"
echo "  GPUs:   ${GPUS} (n=${NUM_PROCESSES})"
echo "  port:   ${MASTER_PORT}"
echo "========================================"

"${ENV_PY}/accelerate" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --run_id "${RUN_ID}"
