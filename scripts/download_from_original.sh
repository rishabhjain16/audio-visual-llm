#!/bin/bash
# Script to set up AVSR-LLM using original codebase checkpoints

# Get the absolute path of the script directory
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
CHECKPOINTS_DIR=${ROOT}/checkpoints

echo "=== Setting up AVSR-LLM with original codebase checkpoints ==="

# Create checkpoints directory if it doesn't exist
mkdir -p ${CHECKPOINTS_DIR}

# Check if AV-HuBERT checkpoint exists
AVHUBERT_CHECKPOINT="${CHECKPOINTS_DIR}/large_vox_iter5.pt"
if [ -f "${AVHUBERT_CHECKPOINT}" ]; then
    echo "✓ Found AV-HuBERT checkpoint at ${AVHUBERT_CHECKPOINT}"
else
    echo "✗ AV-HuBERT checkpoint not found at ${AVHUBERT_CHECKPOINT}"
    echo "  Please copy your AV-HuBERT checkpoint to this location."
    echo "  You can download it from: https://dl.fbaipublicfiles.com/avhubert/models/large_vox_iter5.pt"
fi

# Check if LLM checkpoint exists
LLM_DIR="${CHECKPOINTS_DIR}/Llama-2-7b-hf"
if [ -d "${LLM_DIR}" ] && [ -f "${LLM_DIR}/config.json" ]; then
    echo "✓ Found LLM checkpoint at ${LLM_DIR}"
else
    echo "✗ LLM checkpoint not found at ${LLM_DIR}"
    echo "  Please copy your LLM model files to this directory."
    echo "  Alternatively, you can run: python scripts/download_models.py --llm meta-llama/Llama-2-7b-hf"
fi

# Create symbolic links to dataset if needed
DATA_DIR="/home/rishabh/Desktop/Datasets/lrs3"
if [ -d "${DATA_DIR}" ]; then
    echo "✓ Found dataset directory at ${DATA_DIR}"
else
    echo "✗ Dataset directory not found at ${DATA_DIR}"
    echo "  Please ensure your dataset is at this location or update the paths in the scripts."
fi

echo ""
echo "=== Setup Instructions ==="
echo "1. Make sure you have all required checkpoints in the checkpoints directory."
echo "2. Install all dependencies: pip install -r requirements.txt"
echo "3. Run training: bash scripts/train.sh"
echo "4. Run inference: bash scripts/decode.sh"
echo ""
echo "Note: You may need to adjust paths in scripts/train.sh and scripts/decode.sh"
echo "      if your checkpoints or data are in different locations."
