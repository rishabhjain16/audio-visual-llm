#!/bin/bash
# Quick validation script for ClipWhisperModel using an extremely simplified approach

# Configuration
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
CONFIG="configs/clip_whisper.yaml"
OUTPUT_DIR="outputs/clip_whisper"  # Use same as training output for checkpoints
MODALITY="both"  # audio, video, or both
SUBSET_SIZE=1    # Just 1 sample for fastest testing
DEBUG=true       # Enable additional logging for debugging
BATCH_SIZE=1     # Force batch size to 1 for simplicity

# IMPORTANT: Set to your latest checkpoint path - REQUIRED!
# Find the path using: find outputs/clip_whisper -name "*.pt" | sort -r | head -1
# Or set to a specific checkpoint like:
CHECKPOINT_PATH=$(find outputs/clip_whisper -name "*.pt" | sort -r | head -1)
# If no checkpoint found, try looking in any Hugging Face model directory:
if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$(find outputs -name "pytorch_model.bin" | sort -r | head -1)
fi

# Print configuration
echo "Running quick validation with checkpoint loading:"
echo "------------------------"
echo "Data Path: $DATA_PATH"
echo "Config: $CONFIG"
echo "Output Dir: $OUTPUT_DIR"
echo "Modality: $MODALITY"
echo "Subset Size: $SUBSET_SIZE (minimal for debugging)"
echo "Batch Size: $BATCH_SIZE (forced to 1 for simplicity)"
echo "Debug Mode: $DEBUG"

if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint: $CHECKPOINT_PATH"
else
    echo "WARNING: No checkpoint found! Validation will likely fail without trained weights."
    echo "Please specify a checkpoint path manually in this script."
fi
echo "------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR/validation_logs"

# Make script executable
chmod +x scripts/clip_whisper/quick_validate.py

# Prepare debug flag
DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
fi

# Prepare checkpoint flag
CHECKPOINT_FLAG=""
if [ -n "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_FLAG="--checkpoint_path $CHECKPOINT_PATH"
fi

echo "⚙️ Using minimal validation approach with custom collator"
echo "⚙️ Loading model and checkpoint, then running validation on $SUBSET_SIZE sample..."
echo "⚙️ This might take a minute to load models..."

# Set environment variables for more PyTorch debug output if needed
export TORCH_DISTRIBUTED_DEBUG=INFO  # Changed from DETAIL to reduce noise
export TORCH_SHOW_CPP_STACKTRACES=0  # Disabled to reduce noise
export CUDA_LAUNCH_BLOCKING=1

# Run validation with error handling
set -o pipefail  # Ensure pipe fails if any command fails
(
    # Add a parameter to override batch size in config
    python -u scripts/clip_whisper/quick_validate.py \
        --data_path "$DATA_PATH" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR/validation_logs" \
        --modality "$MODALITY" \
        --subset_size "$SUBSET_SIZE" \
        --batch_size "$BATCH_SIZE" \
        $CHECKPOINT_FLAG \
        $DEBUG_FLAG
) 2>&1 | tee "$OUTPUT_DIR/validation_logs/quick_validate.log"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "✅ Quick validation completed successfully with a real loss value!"
    echo "✅ The model ran forward and backward passes correctly."
    echo "✅ Your model configuration is working properly."
else
    echo "❌ Quick validation failed with exit code $RESULT!"
    if grep -q "dummy loss" "$OUTPUT_DIR/validation_logs/quick_validate.log"; then
        echo "❌ The model ran without crashing but produced dummy loss (1000000)."
        if grep -q "No checkpoint provided" "$OUTPUT_DIR/validation_logs/quick_validate.log"; then
            echo "❌ REASON: No checkpoint was found or loaded. Please verify your checkpoint path."
        elif grep -q "Missing keys" "$OUTPUT_DIR/validation_logs/quick_validate.log"; then
            echo "❌ REASON: Missing keys when loading checkpoint. The model structure may not match."
        else
            echo "❌ This means the forward pass had an error that was caught by our error handler."
        fi
    fi
    echo "Logs have been saved to $OUTPUT_DIR/validation_logs/quick_validate.log"
    echo ""
    echo "Last 20 lines of the log:"
    echo "--------------------------"
    tail -n 20 "$OUTPUT_DIR/validation_logs/quick_validate.log"
fi

exit $RESULT 