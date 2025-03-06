#!/bin/bash
# Decoding script for AVSR-LLM

# Get the absolute path of the script directory
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

# Set variables (using same paths as original codebase)
DATA_PATH=/home/rishabh/Desktop/Datasets/lrs3/433h_data
CHECKPOINT_PATH=${ROOT}/checkpoints/trained/avsr_llm_output/best_model.pt
OUTPUT_DIR=${ROOT}/results

# Configuration file
CONFIG_FILE=${ROOT}/configs/default.yaml

# Language for decoding (usually 'en' for English)
LANG=${1:-"en"}

echo "=== AVSR-LLM Decoding ==="
echo "Data path: ${DATA_PATH}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Language: ${LANG}"

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
mkdir -p ${OUTPUT_DIR}/${LANG}

# Setup environment
export CUDA_VISIBLE_DEVICES=0

# Run decoding
python ${ROOT}/scripts/inference.py \
    --config ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_PATH} \
    --input ${DATA_PATH}/${LANG} \
    --output ${OUTPUT_DIR}/${LANG} \
    --mode av \
    --gpu 0 \
    --beam_size 5

echo "Decoding complete. Results saved to ${OUTPUT_DIR}/${LANG}"
