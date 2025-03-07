#!/bin/bash
# Training script for the ClipWhisperModel

# Default paths
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
OUTPUT_PATH="outputs/clip_whisper"
LLM_PATH="checkpoints/Llama-3.2-1B"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
CONFIG_FILE="configs/clip_whisper.yaml"
GPU_ID=0
DEBUG=false
BATCH_SIZE=2
MODALITY="video"  # Default to using both audio and video
SAVE_EVERY=1
SAVE_STEPS=""
LOG_PARAM_UPDATES=false
MAX_EPOCHS=10

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
echo "=== ClipWhisper Model Training ==="
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
if [ "$LOG_PARAM_UPDATES" = true ]; then
  echo "Logging parameter updates: Yes"
else
  echo "Logging parameter updates: No"
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Start training
echo "Starting training with modality: $MODALITY..."
python_cmd="python scripts/clip_whisper/train.py \
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