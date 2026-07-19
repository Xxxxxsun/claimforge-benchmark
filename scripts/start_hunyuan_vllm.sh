#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${HUNYUAN_VENV_DIR:-/root/HunyuanImage-3.0/.venv}"
MODEL_DIR="${HUNYUAN_MODEL_DIR:-/root/models/HunyuanImage-3-Instruct-Distil}"
DEPLOY_CONFIG="${HUNYUAN_DEPLOY_CONFIG:-/root/claimforge-benchmark/config/hunyuan_image3_8gpu.yaml}"
SERVED_MODEL_NAME="${HUNYUAN_SERVED_MODEL_NAME:-vllm_hunyuan_image3}"
SERVER_HOST="${HUNYUAN_HOST:-127.0.0.1}"
SERVER_PORT="${HUNYUAN_PORT:-8001}"

if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "vLLM executable not found: $VENV_DIR/bin/vllm" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model snapshot is incomplete or missing: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$DEPLOY_CONFIG" ]]; then
  echo "Deploy config not found: $DEPLOY_CONFIG" >&2
  exit 1
fi

# The service is fully local after the snapshot is downloaded. Keeping it
# offline prevents accidental Hub traffic and makes proxy state irrelevant.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${HUNYUAN_NCCL_DEBUG:-WARN}"

exec "$VENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --omni \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --deploy-config "$DEPLOY_CONFIG" \
  "$@"
