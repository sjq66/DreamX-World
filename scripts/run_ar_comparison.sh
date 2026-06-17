#!/usr/bin/env bash
set -euo pipefail

# Run paired AR inference for baseline FIFO KV cache and the experimental
# similarity-based KV eviction policy. Keep all inputs identical except the KV
# eviction policy so the outputs can be compared directly.
#
# Usage:
#   source .venv/bin/activate
#   source dreamx_ar_model.env
#   bash scripts/run_ar_comparison.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/dreamx_ar_model.env" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/dreamx_ar_model.env"
fi

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_PATH="${CONFIG_PATH:-configs/dreamx-ar/causal_camera_forcing_5b.yaml}"
MODEL_NAME="${MODEL_NAME:-${PROJECT_ROOT}/Wan2.2-TI2V-5B}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-./configs/dreamx-ar/}"
DATA_PATH="${DATA_PATH:-configs/dreamx/eval.json}"
FPS="${FPS:-16}"
SEED="${SEED:-42}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
COLOR_CORRECTION_STRENGTH="${COLOR_CORRECTION_STRENGTH:-1.0}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-./outputs_ar_baseline/}"
KV_OUTPUT_DIR="${KV_OUTPUT_DIR:-./outputs_ar_kv_similarity/}"
COMPARISON_REPORT_DIR="${COMPARISON_REPORT_DIR:-./outputs_ar_comparison_report/}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
KV_THRESHOLD="${KV_THRESHOLD:-0.95}"
KV_RECENT_KEEP_CHUNKS="${KV_RECENT_KEEP_CHUNKS:-1}"
KV_SOURCE="${KV_SOURCE:-k}"
CHUNK_RELATIVE="${CHUNK_RELATIVE:---chunk_relative}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

build_common_args() {
  local args=(
    --config_path "${CONFIG_PATH}"
    --model_name "${MODEL_NAME}"
    --transformer_path "${TRANSFORMER_PATH}"
    --data_path "${DATA_PATH}"
    --num_output_frames "${NUM_OUTPUT_FRAMES}"
    --fps "${FPS}"
    --seed "${SEED}"
    --color_correction_strength "${COLOR_CORRECTION_STRENGTH}"
  )

  if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    args+=(--base_checkpoint_path "${BASE_CHECKPOINT_PATH}")
  fi
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    args+=(--checkpoint_path "${CHECKPOINT_PATH}")
  fi
  if [[ -n "${VAE_PATH:-}" ]]; then
    args+=(--vae_path "${VAE_PATH}")
  fi
  if [[ -n "${CHUNK_RELATIVE}" ]]; then
    args+=("${CHUNK_RELATIVE}")
  fi

  printf '%q ' "${args[@]}"
}

COMMON_ARGS="$(build_common_args)"

echo "=============================================="
echo "  AR Comparison"
echo "=============================================="
echo "Python:       ${PYTHON_BIN}"
echo "CUDA devices: ${CUDA_DEVICES}"
echo "Data:         ${DATA_PATH}"
echo "Frames:       ${NUM_OUTPUT_FRAMES} latent frames"
echo "Seed:         ${SEED}"
echo "Baseline out: ${BASELINE_OUTPUT_DIR}"
echo "KV out:       ${KV_OUTPUT_DIR}"
echo "Report out:   ${COMPARISON_REPORT_DIR}"
echo "=============================================="

echo "[1/3] Running baseline FIFO inference"
eval "\"${PYTHON_BIN}\" inference_ar_forcing.py ${COMMON_ARGS} --output_folder \"${BASELINE_OUTPUT_DIR}\" --kv_evict_policy fifo"

echo "[2/3] Running similarity KV inference"
eval "\"${PYTHON_BIN}\" inference_ar_forcing.py ${COMMON_ARGS} --output_folder \"${KV_OUTPUT_DIR}\" --kv_evict_policy similarity --kv_similarity_threshold \"${KV_THRESHOLD}\" --kv_similarity_recent_keep_chunks \"${KV_RECENT_KEEP_CHUNKS}\" --kv_similarity_source \"${KV_SOURCE}\" --kv_similarity_debug"

echo "[3/3] Building comparison report"
"${PYTHON_BIN}" scripts/compare_ar_outputs.py \
  --baseline_dir "${BASELINE_OUTPUT_DIR}" \
  --candidate_dir "${KV_OUTPUT_DIR}" \
  --output_dir "${COMPARISON_REPORT_DIR}" \
  --baseline_label "baseline_fifo" \
  --candidate_label "kv_similarity_t${KV_THRESHOLD}" \
  --side_by_side_scale 0.5

echo "Done. Report: ${COMPARISON_REPORT_DIR}/summary.md"
