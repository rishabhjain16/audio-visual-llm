#!/bin/bash
# Training script for the ClipWhisperModel

# Default values
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
CONFIG="configs/clip_whisper.yaml"
OUTPUT_DIR="outputs/clip_whisper"
LLM_PATH="checkpoints/Llama-2-7b-hf"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
BATCH_SIZE=2
MAX_EPOCHS=5
MODALITY="both"  # audio, video, or both
SAVE_EVERY=1
FP16="false"
USE_4BIT="false"
NO_LORA="false"  # By default, use LoRA
RESUME_FROM=""
DEBUG_MODE="false"  # Debug logging mode
MAX_SEQ_LEN=1536  # Default to 1536 (can be overridden via command line)
LOG_LEVEL="info"  # Default log level
CONNECTOR_TYPE="simple"  # Default to simple connector

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
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
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
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
      LOG_LEVEL="debug"
      shift
      ;;
    --log_level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --max_seq_len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --connector_type)
      CONNECTOR_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
  echo "Error: data_path must be specified"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Print configuration
echo "Training ClipWhisperModel with the following configuration:"
echo "======================"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_DIR"
echo "LLM path: $LLM_PATH"
echo "Whisper model: $WHISPER_MODEL"
echo "CLIP model: $CLIP_MODEL"
echo "Config file: $CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Modality: $MODALITY"
echo "Max epochs: $MAX_EPOCHS"
echo "Save checkpoint every: $SAVE_EVERY epochs"
echo "Logging parameter updates: $([ -n "$LOG_PARAM_UPDATES" ] && echo "Yes" || echo "False")"
echo "Using mixed precision (FP16): $([ "$FP16" == "true" ] && echo "True" || echo "False")"
echo "Using 4-bit quantization: $([ "$USE_4BIT" == "true" ] && echo "True" || echo "False")"
echo "Using LoRA: $([ "$NO_LORA" == "false" ] && echo "Yes" || echo "No")"
echo "Debug mode: ${DEBUG_MODE^}"
echo "Log level: $LOG_LEVEL"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Connector type: $CONNECTOR_TYPE"

# Build command
CMD="python scripts/clip_whisper/train.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --data_path $DATA_PATH \
  --llm_path $LLM_PATH \
  --whisper_model $WHISPER_MODEL \
  --clip_model $CLIP_MODEL \
  --batch_size $BATCH_SIZE \
  --max_epochs $MAX_EPOCHS \
  --modality $MODALITY \
  --save_every $SAVE_EVERY \
  --max_seq_len $MAX_SEQ_LEN \
  --log_level $LOG_LEVEL \
  --connector_type $CONNECTOR_TYPE"

# Add optional arguments
if [ "$FP16" = "true" ]; then
  CMD="$CMD --fp16"
fi

if [ "$USE_4BIT" = "true" ]; then
  CMD="$CMD --use_4bit"
fi

if [ "$NO_LORA" = "true" ]; then
  CMD="$CMD --no_lora"
fi

if [ -n "$LOG_PARAM_UPDATES" ]; then
  CMD="$CMD --log_param_updates"
fi

if [ -n "$RESUME_FROM" ]; then
  CMD="$CMD --resume_from $RESUME_FROM"
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