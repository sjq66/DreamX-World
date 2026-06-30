#!/usr/bin/env bash
set -euo pipefail

# Evaluate whether a smaller content-aware KV cache can approach a larger FIFO
# cache on long revisitation trajectories.
#
# Core comparison:
#   small FIFO  : W36 + FIFO
#   small Ours  : W36 + stride/similarity eviction
#   large FIFO  : W48 + FIFO
#
# Usage:
#   cd /pfs/shijiaqi/DreamX-World
#   source .venv/bin/activate
#   source dreamx_ar_model.env
#   bash scripts/run_revisitation_memory_benchmark.sh

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
BASE_EVAL_PATH="${BASE_EVAL_PATH:-configs/dreamx/eval.json}"
REVISIT_EVAL_PATH="${REVISIT_EVAL_PATH:-configs/dreamx/revisitation_eval.json}"
REVISIT_PATTERN="${REVISIT_PATTERN:-yaw_pitch_loop}"
REVISIT_LIMIT="${REVISIT_LIMIT:-4}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-63}"
FPS="${FPS:-16}"
SEED="${SEED:-42}"
COLOR_CORRECTION_STRENGTH="${COLOR_CORRECTION_STRENGTH:-1.0}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
SMALL_LOCAL_ATTN_SIZE="${SMALL_LOCAL_ATTN_SIZE:-36}"
LARGE_LOCAL_ATTN_SIZE="${LARGE_LOCAL_ATTN_SIZE:-48}"
SINK_SIZE="${SINK_SIZE:-3}"
KV_POLICY="${KV_POLICY:-stride}"
KV_THRESHOLD="${KV_THRESHOLD:-0.90}"
KV_RECENT_KEEP_CHUNKS="${KV_RECENT_KEEP_CHUNKS:-1}"
KV_SOURCE="${KV_SOURCE:-v}"
KV_STRIDE_ANCHOR_CHUNKS="${KV_STRIDE_ANCHOR_CHUNKS:-1}"
KV_PYRAMID_RECENT_KEEP_CHUNKS="${KV_PYRAMID_RECENT_KEEP_CHUNKS:-3}"
KV_PYRAMID_LONG_KEEP_CHUNKS="${KV_PYRAMID_LONG_KEEP_CHUNKS:-4}"
CHUNK_RELATIVE="${CHUNK_RELATIVE:---chunk_relative}"

SMALL_FIFO_DIR="${SMALL_FIFO_DIR:-./outputs_revisit_w${SMALL_LOCAL_ATTN_SIZE}_fifo/}"
SMALL_OURS_DIR="${SMALL_OURS_DIR:-./outputs_revisit_w${SMALL_LOCAL_ATTN_SIZE}_${KV_POLICY}_${KV_SOURCE}_t${KV_THRESHOLD}/}"
LARGE_FIFO_DIR="${LARGE_FIFO_DIR:-./outputs_revisit_w${LARGE_LOCAL_ATTN_SIZE}_fifo/}"
REPORT_DIR="${REPORT_DIR:-./outputs_revisit_memory_report/}"
EVICTION_LOG="${EVICTION_LOG:-${REPORT_DIR}/kv_eviction_w${SMALL_LOCAL_ATTN_SIZE}_${KV_POLICY}_${KV_SOURCE}_t${KV_THRESHOLD}.csv}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

build_common_args() {
  local args=(
    --config_path "${CONFIG_PATH}"
    --model_name "${MODEL_NAME}"
    --transformer_path "${TRANSFORMER_PATH}"
    --data_path "${REVISIT_EVAL_PATH}"
    --num_output_frames "${NUM_OUTPUT_FRAMES}"
    --fps "${FPS}"
    --seed "${SEED}"
    --color_correction_strength "${COLOR_CORRECTION_STRENGTH}"
    --sink_size "${SINK_SIZE}"
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
echo "  Revisitation Memory Benchmark"
echo "=============================================="
echo "Python:             ${PYTHON_BIN}"
echo "CUDA devices:       ${CUDA_DEVICES}"
echo "Base eval:          ${BASE_EVAL_PATH}"
echo "Revisit eval:       ${REVISIT_EVAL_PATH}"
echo "Pattern:            ${REVISIT_PATTERN}"
echo "Items:              ${REVISIT_LIMIT}"
echo "Latent frames:      ${NUM_OUTPUT_FRAMES}"
echo "Small window:       ${SMALL_LOCAL_ATTN_SIZE}"
echo "Large window:       ${LARGE_LOCAL_ATTN_SIZE}"
echo "Sink size:          ${SINK_SIZE}"
echo "KV policy:          ${KV_POLICY}"
echo "KV source/threshold:${KV_SOURCE}/${KV_THRESHOLD}"
echo "Stride anchors:     ${KV_STRIDE_ANCHOR_CHUNKS}"
echo "Pyramid recent/long:${KV_PYRAMID_RECENT_KEEP_CHUNKS}/${KV_PYRAMID_LONG_KEEP_CHUNKS}"
echo "Report:             ${REPORT_DIR}"
echo "=============================================="

"${PYTHON_BIN}" scripts/create_revisitation_eval.py \
  --input "${BASE_EVAL_PATH}" \
  --output "${REVISIT_EVAL_PATH}" \
  --limit "${REVISIT_LIMIT}" \
  --pattern "${REVISIT_PATTERN}"

echo "[1/4] Running small-window FIFO"
eval "\"${PYTHON_BIN}\" inference_ar_forcing.py ${COMMON_ARGS} --local_attn_size \"${SMALL_LOCAL_ATTN_SIZE}\" --output_folder \"${SMALL_FIFO_DIR}\" --kv_evict_policy fifo"

echo "[2/4] Running small-window KV compression"
mkdir -p "${REPORT_DIR}"
rm -f "${EVICTION_LOG}"
eval "\"${PYTHON_BIN}\" inference_ar_forcing.py ${COMMON_ARGS} --local_attn_size \"${SMALL_LOCAL_ATTN_SIZE}\" --output_folder \"${SMALL_OURS_DIR}\" --kv_evict_policy \"${KV_POLICY}\" --kv_similarity_threshold \"${KV_THRESHOLD}\" --kv_similarity_recent_keep_chunks \"${KV_RECENT_KEEP_CHUNKS}\" --kv_similarity_source \"${KV_SOURCE}\" --kv_stride_anchor_chunks \"${KV_STRIDE_ANCHOR_CHUNKS}\" --kv_pyramid_recent_keep_chunks \"${KV_PYRAMID_RECENT_KEEP_CHUNKS}\" --kv_pyramid_long_keep_chunks \"${KV_PYRAMID_LONG_KEEP_CHUNKS}\" --kv_similarity_debug --kv_eviction_log_path \"${EVICTION_LOG}\""

echo "[3/4] Running large-window FIFO"
eval "\"${PYTHON_BIN}\" inference_ar_forcing.py ${COMMON_ARGS} --local_attn_size \"${LARGE_LOCAL_ATTN_SIZE}\" --output_folder \"${LARGE_FIFO_DIR}\" --kv_evict_policy fifo"

echo "[4/4] Building revisitation report"
"${PYTHON_BIN}" scripts/compare_revisitation_outputs.py \
  --run "small_fifo=${SMALL_FIFO_DIR}" \
  --run "small_ours=${SMALL_OURS_DIR}" \
  --run "large_fifo=${LARGE_FIFO_DIR}" \
  --output_dir "${REPORT_DIR}" \
  --small_fifo_label small_fifo \
  --candidate_label small_ours \
  --large_fifo_label large_fifo

echo "Done."
echo "Report:       ${REPORT_DIR}/summary.md"
echo "Metrics:      ${REPORT_DIR}/metrics.csv"
echo "Evict log:    ${EVICTION_LOG}"
