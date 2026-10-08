#!/usr/bin/env bash
set -euo pipefail

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenResidualWorld}"
FREEZE_MODULES="${FREEZE_MODULES:-qwen_vl_interface,vision_encoder}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3.5-0.8B}"
OFFICIAL_CONFIG="${OFFICIAL_CONFIG:-${STARVLA_DIR}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml}"
WORLD_OVERLAY="${WORLD_OVERLAY:-${STARVLA_DIR}/examples/LIBERO_World/train_files/starvla_qwen_residual_world.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_residual_world_all}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./playground/Checkpoints}"
RUN_ID="${RUN_ID:-qwen_residual_world}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
MAX_HORIZON="${MAX_HORIZON:-500}"
BETA_ALPHA="${BETA_ALPHA:-2.5}"
BETA_BETA="${BETA_BETA:-1.0}"
INFERENCE_HORIZON="${INFERENCE_HORIZON:-10000}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
CONFIG_YAML="${CONFIG_YAML:-${OUTPUT_DIR}/config.launch.yaml}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

"${STARVLA_PYTHON}" examples/LIBERO_World/train_files/merge_config.py \
  --base "${OFFICIAL_CONFIG}" \
  --overlay "${WORLD_OVERLAY}" \
  --output "${CONFIG_YAML}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.inference_horizon "${INFERENCE_HORIZON}" \
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.trajectory_max_horizon "${MAX_HORIZON}" \
  --datasets.vla_data.trajectory_beta_alpha "${BETA_ALPHA}" \
  --datasets.vla_data.trajectory_beta_beta "${BETA_BETA}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
