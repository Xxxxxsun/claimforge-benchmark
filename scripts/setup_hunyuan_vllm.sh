#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${HUNYUAN_VENV_DIR:-/root/HunyuanImage-3.0/.venv}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-/root/vllm-omni}"
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VLLM_OMNI_REF="${VLLM_OMNI_REF:-v0.24.1}"
DISTIL_PATCH="${HUNYUAN_DISTIL_PATCH:-$REPO_DIR/patches/vllm-omni-v0.24.1-hunyuan-image3-distil.patch}"
PYTHON_BIN="${HUNYUAN_PYTHON_BIN:-python3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found in PATH" >&2
  exit 1
fi
if [[ ! -e "$VLLM_OMNI_DIR" ]]; then
  git clone --depth 1 --branch "$VLLM_OMNI_REF" \
    "$VLLM_OMNI_REPO" "$VLLM_OMNI_DIR"
fi
if [[ ! -f "$VLLM_OMNI_DIR/pyproject.toml" ]]; then
  echo "vLLM-Omni checkout not found: $VLLM_OMNI_DIR" >&2
  exit 1
fi
if [[ ! -f "$DISTIL_PATCH" ]]; then
  echo "Distil compatibility patch not found: $DISTIL_PATCH" >&2
  exit 1
fi

# Apply the v0.24.1 Distil/MeanFlow compatibility patch exactly once. Reverse
# checking first makes this safe to rerun against an already patched checkout.
if git -C "$VLLM_OMNI_DIR" apply --reverse --check "$DISTIL_PATCH"; then
  echo "vLLM-Omni Distil patch is already applied"
elif git -C "$VLLM_OMNI_DIR" apply --check "$DISTIL_PATCH"; then
  git -C "$VLLM_OMNI_DIR" apply "$DISTIL_PATCH"
  echo "Applied vLLM-Omni Distil patch"
else
  echo "Cannot apply Distil patch cleanly to $VLLM_OMNI_DIR; expected $VLLM_OMNI_REF" >&2
  exit 1
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

# vLLM-Omni v0.24.1 is paired with vLLM 0.24.0. Let uv select the
# CUDA-compatible PyTorch wheel, then install the local Distil-patched Omni
# checkout in editable mode.
uv pip install --python "$VENV_DIR/bin/python" \
  "vllm==0.24.0" --torch-backend=auto
uv pip install --python "$VENV_DIR/bin/python" \
  --editable "$VLLM_OMNI_DIR"

"$VENV_DIR/bin/python" -c \
  'import torch, vllm, vllm_omni; print(f"torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__} vllm_omni={vllm_omni.__version__}")'
