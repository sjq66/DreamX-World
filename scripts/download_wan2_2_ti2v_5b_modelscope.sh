#!/usr/bin/env bash
set -euo pipefail

# Download Wan2.2-TI2V-5B base weights from ModelScope into the path expected
# by inference_ar_forcing.sh:
#
#   MODEL_NAME=./Wan2.2-TI2V-5B
#
# This base model directory is used for the text encoder, tokenizer, and VAE.
#
# Usage:
#   bash scripts/download_wan2_2_ti2v_5b_modelscope.sh

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/Wan2.2-TI2V-5B}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
UV_BIN="${UV_BIN:-/root/.local/bin/uv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: Python virtualenv not found at ${VENV_DIR}" >&2
  echo "Create it first, or pass VENV_DIR=/path/to/.venv." >&2
  exit 1
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

echo "=============================================="
echo "  Download Wan2.2-TI2V-5B from ModelScope"
echo "=============================================="
echo "Model id:   ${MODEL_ID}"
echo "Model dir:  ${MODEL_DIR}"
echo "Venv:       ${VENV_DIR}"
echo "Python:     ${PYTHON_BIN}"
echo "uv:         ${UV_BIN}"
echo "=============================================="

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import modelscope
PY
then
  if [[ ! -x "${UV_BIN}" ]]; then
    echo "ERROR: modelscope is not installed and uv was not found at ${UV_BIN}" >&2
    echo "Install manually with:" >&2
    echo "  ${PYTHON_BIN} -m pip install -U modelscope" >&2
    exit 1
  fi
  echo "Installing modelscope into ${VENV_DIR} ..."
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" -U modelscope
fi

mkdir -p "${MODEL_DIR}"

"${PYTHON_BIN}" - <<PY
import os
from modelscope import snapshot_download

model_id = os.environ.get("MODEL_ID", "${MODEL_ID}")
model_dir = os.environ.get("MODEL_DIR", "${MODEL_DIR}")

snapshot_download(
    model_id=model_id,
    local_dir=model_dir,
)
print(f"Downloaded {model_id} to {model_dir}")
PY

missing=0
for required in \
  "${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth" \
  "${MODEL_DIR}/Wan2.2_VAE.pth" \
  "${MODEL_DIR}/google/umt5-xxl"; do
  if [[ ! -e "${required}" ]]; then
    echo "WARNING: expected file or directory not found: ${required}" >&2
    missing=1
  fi
done

echo "=============================================="
echo "  Done"
echo "=============================================="
echo "Wan base dir: ${MODEL_DIR}"
if [[ "${missing}" -eq 0 ]]; then
  echo "Required AR base-model files are present."
else
  echo "Download completed, but one or more expected files were not found."
  echo "Please inspect ${MODEL_DIR} before running inference."
fi
echo
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export MODEL_NAME=\"${MODEL_DIR}\""
echo "  source ${PROJECT_ROOT}/dreamx_ar_model.env"
echo "  bash inference_ar_forcing.sh"
