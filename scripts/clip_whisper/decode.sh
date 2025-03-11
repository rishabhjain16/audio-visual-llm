#!/bin/bash
# Decoding script for the ClipWhisperModel

# Default values
MODEL_PATH="outputs/test_clip_whisper/best_model/model"
TEST_DATA="/home/rishabh/Desktop/Datasets/lrs3/433h_data/test.tsv"
TEST_WRD="/home/rishabh/Desktop/Datasets/lrs3/433h_data/test.wrd"
OUTPUT_DIR="outputs/clip_whisper_decoding_infer"
MODALITY="video"
BATCH_SIZE=1
MAX_NEW_TOKENS=100
DEVICE="cuda"
CONFIG="configs/clip_whisper.yaml"
SINGLE_FILE=""
VERBOSE=false
WHISPER_MODEL="checkpoints/whisper-medium"
CLIP_MODEL="checkpoints/clip-vit-base-patch32"
LLM_MODEL="checkpoints/Llama-3.2-1B"

# Display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --model_path PATH       Path to the trained model directory (required)"
    echo "  --test_data PATH        Path to test data TSV file (required unless --single_file is specified)"
    echo "  --test_wrd PATH         Path to test word reference file (required unless --single_file is specified)"
    echo "  --output_dir DIR        Directory to save decoding results (default: outputs/clip_whisper_decoding)"
    echo "  --modality MODE         Modality to use: audio, video, or both (default: both)"
    echo "  --batch_size N          Batch size for inference (default: 1)"
    echo "  --max_new_tokens N      Maximum number of tokens to generate (default: 100)"
    echo "  --device DEVICE         Device to run inference on (default: cuda)"
    echo "  --config PATH           Configuration file for processor settings (default: configs/clip_whisper.yaml)"
    echo "  --single_file PATH      Path to a single audio/video file for testing"
    echo "  --whisper_model PATH    Path to pre-trained Whisper model (default: checkpoints/whisper-medium)"
    echo "  --clip_model PATH       Path to pre-trained CLIP model (default: checkpoints/clip-vit-base-patch32)"
    echo "  --llm_model PATH        Path to pre-trained LLM model (default: checkpoints/Llama-3.2-1B)"
    echo "  --verbose               Enable verbose output"
    echo "  -h, --help              Display this help message and exit"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --test_wrd)
            TEST_WRD="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --single_file)
            SINGLE_FILE="$2"
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
        --llm_model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    show_help
    exit 1
fi

if [ -z "$SINGLE_FILE" ] && ([ -z "$TEST_DATA" ] || [ -z "$TEST_WRD" ]); then
    echo "Error: Both --test_data and --test_wrd are required unless --single_file is specified"
    show_help
    exit 1
fi

# Construct the command to run the Python script
CMD="python scripts/clip_whisper/decode.py --model_path $MODEL_PATH"

if [ -n "$SINGLE_FILE" ]; then
    CMD+=" --single_file $SINGLE_FILE"
else
    CMD+=" --test_data $TEST_DATA --test_wrd $TEST_WRD"
fi

CMD+=" --output_dir $OUTPUT_DIR"
CMD+=" --modality $MODALITY"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --max_new_tokens $MAX_NEW_TOKENS"
CMD+=" --device $DEVICE"
CMD+=" --config $CONFIG"
CMD+=" --whisper_model $WHISPER_MODEL"
CMD+=" --clip_model $CLIP_MODEL"
CMD+=" --llm_model $LLM_MODEL"
CMD+=" --text_key text"
CMD+=" --calculate_loss"

if [ "$VERBOSE" = true ]; then
    CMD+=" --verbose"
fi

# Print command summary
echo "=== ClipWhisper Decoding ==="
echo "Model path: $MODEL_PATH"
if [ -n "$SINGLE_FILE" ]; then
    echo "Single file: $SINGLE_FILE"
else
    echo "Test data: $TEST_DATA"
    echo "Test WRD: $TEST_WRD"
fi
echo "Modality: $MODALITY"
echo "Output directory: $OUTPUT_DIR"

# Run the command
echo -e "\nRunning command: $CMD"
eval $CMD 