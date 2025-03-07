#!/bin/bash
# Training script for AVSR-LLM model

# Default paths
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
OUTPUT_PATH="checkpoints/trained/avsr_llm_output"
AV_HUBERT_PATH="checkpoints/large_vox_iter5.pt"
LLM_PATH="checkpoints/Llama-3.2-1B"
WHISPER_MODEL="openai/whisper-medium"
CONFIG_FILE="configs/default.yaml"
GPU_ID=0
DEBUG=false

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
    --av_hubert_path)
      AV_HUBERT_PATH="$2"
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
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --debug)
      DEBUG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== AVSR-LLM Training ==="
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "AV-HuBERT path: $AV_HUBERT_PATH"
echo "LLM path: $LLM_PATH"
echo "Whisper model: $WHISPER_MODEL"
echo "Config file: $CONFIG_FILE"
echo "Debug mode: $DEBUG"

# Check paths exist
if [ -f "$AV_HUBERT_PATH" ]; then
  echo "Using existing AV-HuBERT model: $AV_HUBERT_PATH"
else
  echo "Error: AV-HuBERT model not found at $AV_HUBERT_PATH"
  exit 1
fi

if [ -d "$LLM_PATH" ]; then
  echo "Using existing LLM model: $LLM_PATH"
else
  echo "Error: LLM model not found at $LLM_PATH"
  exit 1
fi

# Ensure Whisper model is available
WHISPER_CACHE="checkpoints/whisper/whisper-medium"
echo "Ensuring Whisper model is available: $WHISPER_MODEL"
if [ -d "$WHISPER_CACHE" ]; then
  echo "Using cached Whisper model from: $WHISPER_CACHE"
else
  echo "Downloading Whisper model to $WHISPER_CACHE"
  # Create directory
  mkdir -p "checkpoints/whisper"
  # Download model (it will be saved to the checkpoints directory by the WhisperEncoder)
  python -c "from transformers import WhisperModel, WhisperProcessor; model = WhisperModel.from_pretrained('$WHISPER_MODEL'); processor = WhisperProcessor.from_pretrained('$WHISPER_MODEL'); model.save_pretrained('$WHISPER_CACHE'); processor.save_pretrained('$WHISPER_CACHE'); print('Whisper model downloaded and saved')"
fi
echo "Whisper model ready."

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Start training
echo "Starting training..."
python scripts/train.py \
  --config "$CONFIG_FILE" \
  --checkpoint_dir "$OUTPUT_PATH" \
  --data_path "$DATA_PATH" \
  --llm_path "$LLM_PATH" \
  --av_encoder_path "$AV_HUBERT_PATH" \
  --gpu "$GPU_ID" \
  --seed 42
