#!/usr/bin/env bash
# Memory-only training using the same trainer and LIBERO data as the V2 run.
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
V2_CKPT="${V2_CKPT:?set V2_CKPT to the trained V2 weight file}"
V2_CONFIG="${V2_CONFIG:?set V2_CONFIG to the matching config.full.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/playground/Checkpoints/inverse_v3_ttt/config.launch.yaml}"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
"${STARVLA_PYTHON}" examples/LIBERO_World/train_files/prepare_inverse_v3_config.py \
  --base "${V2_CONFIG}" --checkpoint "${V2_CKPT}" --output "${CONFIG_YAML}"
"${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  examples/LIBERO_World/train_files/train_inverse_v3_ttt.py --config_yaml "${CONFIG_YAML}" "$@"
