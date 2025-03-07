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
BATCH_SIZE=4  # Default to a small batch size for stability
MODALITY="audio"  # Default to using both audio and video
SAVE_EVERY=1  # Default to saving every epoch
SAVE_STEPS=""  # Empty by default, won't be used unless specified
LOG_PARAM_UPDATES=false  # Default to not logging parameter updates
MAX_EPOCHS=10  # Default number of epochs

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
    --modality)
      MODALITY="$2"
      shift 2
      ;;
    --save_every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --log_param_updates)
      LOG_PARAM_UPDATES=true
      shift
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
echo "Modality: $MODALITY"
echo "Max epochs: $MAX_EPOCHS"
echo "Save every: $SAVE_EVERY epochs"
if [ -n "$SAVE_STEPS" ]; then
  echo "Save every: $SAVE_STEPS steps"
fi

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
  --max_epochs $MAX_EPOCHS \
  --modality $MODALITY \
  --save_every $SAVE_EVERY"

# Add save_steps if specified
if [ -n "$SAVE_STEPS" ]; then
  python_cmd="$python_cmd --save_steps $SAVE_STEPS"
fi

# Add log_param_updates if enabled
if [ "$LOG_PARAM_UPDATES" = true ]; then
  python_cmd="$python_cmd --log_param_updates"
fi

# Add debug flag if enabled
if [ "$DEBUG" = true ]; then
  python_cmd="$python_cmd --debug"
fi

# Run the command
echo "Executing: $python_cmd"
eval $python_cmd

# Check if training completed successfully
if [ $? -eq 0 ]; then
  echo "Training completed successfully"
else
  echo "Training failed with exit code $?"
fi 