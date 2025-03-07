#!/bin/bash
# Training script for the ClipWhisperModel

# Default values
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
OUTPUT_PATH="outputs/clip_whisper"
LLM_PATH="checkpoints/Llama-3.2-1B"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
CONFIG_FILE="configs/clip_whisper.yaml"
BATCH_SIZE=2
MODALITY="video"
MAX_EPOCHS=10
SAVE_EVERY=1
LOG_PARAMS="false"
FP16="false"   # Default is off
USE_4BIT="false"  # Default is off
NO_LORA="false"
RESUME_FROM=""
DEBUG_MODE="false"  # Debug logging mode
MAX_SEQ_LEN=1536  # Default to 1536 (can be overridden via command line)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_dir)
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
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --modality)
      MODALITY="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --save_every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --log_params)
      LOG_PARAMS="true"
      shift
      ;;
    --fp16)
      FP16="true"
      shift
      ;;
    --use_4bit)
      USE_4BIT="true"
      shift
      ;;
    --no_lora)
      NO_LORA="true"
      shift
      ;;
    --resume_from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --debug)
      DEBUG_MODE="true"
      shift
      ;;
    --max_seq_len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "Training ClipWhisperModel with the following configuration:"
echo "======================"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "LLM path: $LLM_PATH"
echo "Whisper model: $WHISPER_MODEL"
echo "CLIP model: $CLIP_MODEL"
echo "Config file: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Modality: $MODALITY"
echo "Max epochs: $MAX_EPOCHS"
echo "Save checkpoint every: $SAVE_EVERY epochs"
echo "Logging parameter updates: ${LOG_PARAMS^}"
echo "Using mixed precision (FP16): ${FP16^}"
echo "Using 4-bit quantization: ${USE_4BIT^}"
echo "Using LoRA: $([ "$NO_LORA" == "false" ] && echo "Yes" || echo "No")"
echo "Debug mode: ${DEBUG_MODE^}"
echo "Max sequence length: $MAX_SEQ_LEN"

# Build command
CMD="python scripts/clip_whisper/train.py \
  --config $CONFIG_FILE \
  --output_dir $OUTPUT_PATH \
  --data_path $DATA_PATH \
  --llm_path $LLM_PATH \
  --whisper_model $WHISPER_MODEL \
  --clip_model $CLIP_MODEL \
  --batch_size $BATCH_SIZE \
  --max_epochs $MAX_EPOCHS \
  --modality $MODALITY \
  --save_every $SAVE_EVERY \
  --max_seq_len $MAX_SEQ_LEN"

# Add optional arguments
if [ "$LOG_PARAMS" = "true" ]; then
  CMD="$CMD --log_param_updates"
fi

if [ "$FP16" = "true" ]; then
  CMD="$CMD --fp16"
fi

if [ "$USE_4BIT" = "true" ]; then
  CMD="$CMD --use_4bit"
fi

if [ "$NO_LORA" = "true" ]; then
  CMD="$CMD --no_lora"
fi

if [ ! -z "$RESUME_FROM" ]; then
  CMD="$CMD --resume_from $RESUME_FROM"
fi

if [ "$DEBUG_MODE" = "true" ]; then
  CMD="$CMD --log_level debug"
fi

echo "Starting training with modality: $MODALITY..."
echo "Executing: $CMD"
$CMD

# Check if training was successful
if [ $? -ne 0 ]; then
  echo "Training failed with exit code $?"
  exit 1
fi

echo "Training completed successfully!" 