#!/bin/bash
# Script for running inference on a single video file

# Get the absolute path of the script directory
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

# Parse arguments
VIDEO_PATH=""
AUDIO_PATH=""
MODE="av"  # Default: use both audio and video
OUTPUT_DIR="${ROOT}/results/single_video"
CHECKPOINT_PATH="${ROOT}/checkpoints/trained/avsr_llm_output/best_model.pt"
CONFIG_FILE="${ROOT}/configs/default.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --video)
      VIDEO_PATH="$2"
      shift 2
      ;;
    --audio)
      AUDIO_PATH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --video PATH       Path to input video file"
      echo "  --audio PATH       Path to input audio file (optional)"
      echo "  --mode MODE        Input mode: 'audio', 'video', or 'av' (default: av)"
      echo "  --output DIR       Output directory (default: results/single_video)"
      echo "  --checkpoint PATH  Path to model checkpoint"
      echo "  --config PATH      Path to configuration file"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Verify inputs
if [[ "$MODE" == "av" && -z "$VIDEO_PATH" ]]; then
    echo "Error: Video path is required for AV mode"
    exit 1
fi

if [[ "$MODE" == "audio" && -z "$AUDIO_PATH" ]]; then
    echo "Error: Audio path is required for audio mode"
    exit 1
fi

if [[ "$MODE" == "video" && -z "$VIDEO_PATH" ]]; then
    echo "Error: Video path is required for video mode"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Warning: Model checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Make sure you've trained the model first or specified the correct path"
    
    # Try to find the best model in the checkpoint directory
    CHECKPOINT_DIR=$(dirname "${CHECKPOINT_PATH}")
    if [ -d "${CHECKPOINT_DIR}" ]; then
        # Look for any checkpoint file
        AVAILABLE_CHECKPOINTS=$(find "${CHECKPOINT_DIR}" -name "*.pt" | sort)
        if [ -n "${AVAILABLE_CHECKPOINTS}" ]; then
            # Use the first available checkpoint
            CHECKPOINT_PATH=$(echo "${AVAILABLE_CHECKPOINTS}" | head -n 1)
            echo "Using alternative checkpoint: ${CHECKPOINT_PATH}"
        else
            echo "No checkpoints found in ${CHECKPOINT_DIR}"
            exit 1
        fi
    else
        echo "Checkpoint directory doesn't exist"
        exit 1
    fi
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Set up environment
export CUDA_VISIBLE_DEVICES=0

echo "=== AVSR-LLM Single Video Inference ==="
echo "Mode: ${MODE}"
if [[ -n "$VIDEO_PATH" ]]; then
    echo "Video: ${VIDEO_PATH}"
fi
if [[ -n "$AUDIO_PATH" ]]; then
    echo "Audio: ${AUDIO_PATH}"
fi
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Build command based on mode
CMD="python ${ROOT}/scripts/inference.py \
    --config ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output ${OUTPUT_DIR} \
    --mode ${MODE} \
    --gpu 0"

if [[ -n "$VIDEO_PATH" ]]; then
    CMD="${CMD} --input ${VIDEO_PATH}"
elif [[ -n "$AUDIO_PATH" ]]; then
    CMD="${CMD} --input ${AUDIO_PATH}"
fi

# Run inference
eval "${CMD}"

echo "Inference complete. Results saved to ${OUTPUT_DIR}"
echo "Check ${OUTPUT_DIR}/transcriptions.json for results"
