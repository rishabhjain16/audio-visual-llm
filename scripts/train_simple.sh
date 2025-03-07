#!/bin/bash
# Simplified training script for AVSR-LLM model

# Default paths
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
OUTPUT_PATH="outputs/simple_avsr"
LLM_PATH="checkpoints/Llama-3.2-1B"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
CONFIG_FILE="configs/simple.yaml"
GPU_ID=0
DEBUG=false
BATCH_SIZE=2  # Default to a small batch size for stability

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --llm_path)
      LLM_PATH="$2"
      shift 2
      ;;
    --whisper_model)
      WHISPER_MODEL="$2"
      shift 2
      ;;
    --clip_model)
      CLIP_MODEL="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== Simple AVSR-LLM Training ==="
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "LLM path: $LLM_PATH"
echo "Whisper model: $WHISPER_MODEL"
echo "CLIP model: $CLIP_MODEL"
echo "Config file: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo "Batch size: $BATCH_SIZE"

# Check paths exist
if [ -d "$LLM_PATH" ]; then
  echo "Using existing LLM model: $LLM_PATH"
else
  echo "Error: LLM model not found at $LLM_PATH"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Start training directly without using conda activate
echo "Starting training..."
python_cmd="python scripts/train_simple.py \
  --config $CONFIG_FILE \
  --output_dir $OUTPUT_PATH \
  --data_path $DATA_PATH \
  --llm_path $LLM_PATH \
  --whisper_model $WHISPER_MODEL \
  --clip_model $CLIP_MODEL \
  --gpu $GPU_ID \
  --batch_size $BATCH_SIZE \
  --fp16"

# Add debug flag if enabled
if [ "$DEBUG" = true ]; then
  python_cmd="$python_cmd --debug"
fi

# Run the command
eval $python_cmd

# Check if training completed successfully
if [ $? -eq 0 ]; then
  echo "Training completed successfully"
else
  echo "Training failed with exit code $?"
fi 