#!/bin/bash
# Script to verify fairseq installation and prepare for AV-HuBERT loading

# Set paths
ROOT_DIR=$(pwd)
FAIRSEQ_DIR="${ROOT_DIR}/fairseq"

echo "=== Checking Fairseq Installation ==="
echo "Root directory: ${ROOT_DIR}"
echo "Fairseq directory: ${FAIRSEQ_DIR}"

# Check if fairseq directory exists
if [ -d "${FAIRSEQ_DIR}" ]; then
    echo "✅ Fairseq directory found"
    
    # Check if fairseq is installed in Python
    if python -c "import fairseq; print(f'Fairseq version: {fairseq.__version__}')" 2>/dev/null; then
        echo "✅ Fairseq is importable in Python"
    else
        echo "❌ Fairseq is not importable. Installing fairseq in development mode..."
        # Try to install fairseq
        cd "${FAIRSEQ_DIR}"
        pip install -e .
        cd "${ROOT_DIR}"
        
        # Check again
        if python -c "import fairseq; print(f'Fairseq version: {fairseq.__version__}')" 2>/dev/null; then
            echo "✅ Fairseq installed successfully"
        else
            echo "❌ Failed to install fairseq. AV-HuBERT loading may not work correctly."
        fi
    fi
else
    echo "❌ Fairseq directory not found at ${FAIRSEQ_DIR}"
    echo "Please make sure you have fairseq in your project directory."
fi

# Check omegaconf version
OMEGACONF_VERSION=$(pip list | grep omegaconf | awk '{print $2}')
echo "Omegaconf version: ${OMEGACONF_VERSION}"

if [[ $(echo "${OMEGACONF_VERSION}" | awk -F. '{ print $1 }') -ge 2 ]]; then
    echo "✅ Omegaconf version is compatible (≥ 2.0)"
else
    echo "❌ Omegaconf version is too old. Upgrading to latest version..."
    pip install -U omegaconf>=2.0.0
    
    # Check again
    OMEGACONF_VERSION=$(pip list | grep omegaconf | awk '{print $2}')
    echo "Updated omegaconf version: ${OMEGACONF_VERSION}"
fi

# Check AV-HuBERT checkpoint
CHECKPOINT_PATH="checkpoints/large_vox_iter5.pt"

if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "✅ AV-HuBERT checkpoint found at ${CHECKPOINT_PATH}"
    # Get file size in MB
    SIZE_MB=$(du -m "${CHECKPOINT_PATH}" | cut -f1)
    echo "   - Checkpoint size: ${SIZE_MB} MB"
    
    # A healthy AV-HuBERT checkpoint should be at least 100MB
    if [ "${SIZE_MB}" -lt 100 ]; then
        echo "⚠️  Warning: Checkpoint file seems too small. It might be incomplete."
    fi
else
    echo "❌ AV-HuBERT checkpoint not found at ${CHECKPOINT_PATH}"
fi

echo ""
echo "Now testing AV-HuBERT model loading with Python script..."

# Run the verification script
python scripts/verify_avhubert.py --checkpoint "${CHECKPOINT_PATH}" --use_video

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ AV-HuBERT model verification completed successfully."
else
    echo "❌ AV-HuBERT model verification failed with exit code ${EXIT_CODE}."
fi 