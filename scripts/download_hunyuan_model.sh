#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${HUNYUAN_MODEL_REPO:-tencent/HunyuanImage-3.0-Instruct-Distil}"
MODEL_REVISION="${HUNYUAN_MODEL_REVISION:-c8ffd07206f1b843697606968196e8f59f8ff38c}"
MODEL_DIR="${HUNYUAN_MODEL_DIR:-/root/models/HunyuanImage-3-Instruct-Distil}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

mkdir -p "$MODEL_DIR"

# Deliberately remove both upper- and lower-case proxy variables. The endpoint
# remains explicit so this command is safe to rerun to resume an interrupted
# snapshot download.
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT="$HF_MIRROR" \
  HF_HUB_DOWNLOAD_TIMEOUT=600 \
  HF_HUB_ETAG_TIMEOUT=60 \
  hf download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR" \
    --max-workers "$HF_MAX_WORKERS"
