#!/bin/bash
# Simplified training script for ClipWhisperModel

# Training Configuration
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
CONFIG="configs/clip_whisper.yaml"
OUTPUT_DIR="outputs/clip_whisper_new_adaptive_test_ran"
LLM_PATH="checkpoints/Llama-2-7b-hf"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
BATCH_SIZE=2
MAX_EPOCHS=10
MODALITY="both"  # audio, video, or both
SAVE_EVERY=1
FP16=true
USE_4BIT=true
NO_LORA=false
MAX_SEQ_LEN=1536
CONNECTOR_TYPE="simple"
MAX_GRAD_NORM=0.5
LEARNING_RATE="1e-4"

# Print configuration
echo "Starting training with the following configuration:"
echo "-----------------------------------------------"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Epochs: $MAX_EPOCHS"
echo "Modality: $MODALITY"
echo "FP16: $FP16"
echo "4-bit Quantization: $USE_4BIT"
echo "LoRA: $([ "$NO_LORA" = "true" ] && echo "Disabled" || echo "Enabled")"
echo "-----------------------------------------------"

# Prepare boolean flags
FP16_FLAG=""
if [ "$FP16" = "true" ]; then
    FP16_FLAG="--fp16"
fi

USE_4BIT_FLAG=""
if [ "$USE_4BIT" = "true" ]; then
    USE_4BIT_FLAG="--use_4bit"
fi

NO_LORA_FLAG=""
if [ "$NO_LORA" = "true" ]; then
    NO_LORA_FLAG="--no_lora"
fi

# Run training
python scripts/clip_whisper/train.py \
    --data_path "$DATA_PATH" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --llm_path "$LLM_PATH" \
    --whisper_model "$WHISPER_MODEL" \
    --clip_model "$CLIP_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --modality "$MODALITY" \
    --save_every "$SAVE_EVERY" \
    $FP16_FLAG \
    $USE_4BIT_FLAG \
    $NO_LORA_FLAG \
    --max_seq_len "$MAX_SEQ_LEN" \
    --connector_type "$CONNECTOR_TYPE" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --learning_rate "$LEARNING_RATE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed. Check the logs for more information."
    exit 1
fi 