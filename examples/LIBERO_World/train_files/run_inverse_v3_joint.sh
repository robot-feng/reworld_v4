#!/usr/bin/env bash
# Train V3 backbone and feedback memory together in one optimizer run.
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/miniconda3/envs/ResWAM/bin/python}"
MASTER_PORT="${MASTER_PORT:-29864}"
CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO_World/train_files/inverse_v3_h8_h64_joint.yaml}"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eno1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
"${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_v3_accum2.yaml \
  --num_processes 4 --main_process_port "${MASTER_PORT}" \
  examples/LIBERO_World/train_files/train_inverse_v3_ttt.py \
  --config_yaml "${CONFIG_YAML}" "$@"
