#!/bin/bash
# ============================================================================
# AR-Forcing: Chunk-wise causal video generation with camera control
# ============================================================================
# Usage:
#   bash inference_ar_forcing.sh
#
# Override defaults via environment variables:
#   BASE_CHECKPOINT_PATH=/path/to/baseline.pt bash inference_ar_forcing.sh
# ============================================================================

# ======================== Model Path ========================
MODEL_NAME="${MODEL_NAME:-./Wan2.2-TI2V-5B}"   # Path to the folder containing Wan2.2 base model weights (text encoder, tokenizer, VAE).
CONFIG_PATH="configs/dreamx-ar/causal_camera_forcing_5b.yaml"  # Path to AR-forcing YAML config file.
TRANSFORMER_PATH="./configs/dreamx-ar/"  # Path to the folder containing AR-forcing model config.json.
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH-./DreamX-World-5B/baseline.pt}"  # Path to base .pt checkpoint.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Optional additional .pt/.safetensors checkpoint or checkpoint directory.
# VAE_PATH=""                    # (Optional) Path to VAE checkpoint, overrides MODEL_NAME/Wan2.2_VAE.pth.

# ====================== Basic settings ======================
DATA_PATH="${DATA_PATH:-configs/dreamx/eval.json}" # Path to input JSON file
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs_ar/}" # Path to save output video
NUM_OUTPUT_FRAMES=21             # Latent frames, shall be divisible by 3. Pixel frames = (N-1)*4+1. 21→81 pixels (5s@16fps), 63→249 pixels
FPS=16                           # FPS of output video
SEED=42                          # Random seed

# ====================== Post-processing ======================
COLOR_CORRECTION_STRENGTH=1.0    # Lab color correction (0=off, 1=full)
TEMPORAL_SMOOTHING_WINDOW=1      # Temporal Gaussian smoothing (1=off)
BLEND_OVERLAP_FRAMES=6           # Chunk boundary blending frames
CHUNK_RELATIVE="--chunk_relative"  # Per-chunk relative poses (recommended for long videos)

# ====================== GPU ======================
CUDA_DEVICES="${CUDA_DEVICES:-0}"

# ======================== Build Command ========================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "Using GPU: ${CUDA_DEVICES}"

CMD="python inference_ar_forcing.py"
CMD="${CMD} --config_path ${CONFIG_PATH}"
CMD="${CMD} --model_name ${MODEL_NAME}"
CMD="${CMD} --transformer_path ${TRANSFORMER_PATH}"
if [ -n "${BASE_CHECKPOINT_PATH}" ]; then
    CMD="${CMD} --base_checkpoint_path ${BASE_CHECKPOINT_PATH}"
fi
if [ -n "${CHECKPOINT_PATH}" ]; then
    CMD="${CMD} --checkpoint_path ${CHECKPOINT_PATH}"
fi
if [ -n "${VAE_PATH}" ]; then
    CMD="${CMD} --vae_path ${VAE_PATH}"
fi
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --output_folder ${OUTPUT_FOLDER}"
CMD="${CMD} --num_output_frames ${NUM_OUTPUT_FRAMES}"
CMD="${CMD} --fps ${FPS}"
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --color_correction_strength ${COLOR_CORRECTION_STRENGTH}"
CMD="${CMD} ${CHUNK_RELATIVE}"

echo "=============================================="
echo "  AR-Forcing Inference"
echo "=============================================="
echo "Command: ${CMD}"
echo "=============================================="

eval ${CMD}
