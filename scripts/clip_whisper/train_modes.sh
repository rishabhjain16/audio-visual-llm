#!/bin/bash
# Helper script to run training with different memory optimization modes

# Default values
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
EPOCHS=10
BATCH_SIZE=2
MODALITY="both"
MODE="max"  # standard, fp16, 4bit, or max
OUTPUT_DIR="outputs/clip_whisper"
DEBUG="false"
MAX_SEQ_LEN=256  # Default is now 1536 to capture more of the 1500 audio frames

# ANSI color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
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
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --debug)
      DEBUG="true"
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

# Base command
BASE_CMD="bash scripts/clip_whisper/train.sh --data_path $DATA_PATH --max_epochs $EPOCHS --batch_size $BATCH_SIZE --modality $MODALITY --output_dir $OUTPUT_DIR --max_seq_len $MAX_SEQ_LEN"

if [ "$DEBUG" = "true" ]; then
  BASE_CMD="$BASE_CMD --debug"
  echo -e "${YELLOW}Debug mode enabled - you'll see more detailed logs${NC}"
fi

# Show memory usage information based on mode
print_memory_info() {
  echo -e "${BLUE}===============================================${NC}"
  echo -e "${CYAN}              MEMORY USAGE GUIDE              ${NC}"
  echo -e "${BLUE}===============================================${NC}"
  
  case $1 in
    standard)
      echo -e "${RED}STANDARD MODE - HIGHEST MEMORY USAGE${NC}"
      echo -e "• Uses full precision (FP32) for all operations"
      echo -e "• No memory optimizations enabled"
      echo -e "• Recommended only if you have ${RED}plenty of GPU memory${NC}"
      echo -e "• Estimated memory usage: ${RED}100%${NC} of baseline"
      ;;
    fp16)
      echo -e "${YELLOW}FP16 MODE - REDUCED MEMORY USAGE${NC}"
      echo -e "• Uses mixed precision training (FP16)"
      echo -e "• Reduces memory by ~40% with minimal accuracy impact"
      echo -e "• Calculations use FP16, gradients accumulated in FP32"
      echo -e "• Estimated memory usage: ${YELLOW}~60%${NC} of baseline"
      ;;
    4bit)
      echo -e "${GREEN}4-BIT MODE - GREATLY REDUCED MEMORY USAGE${NC}"
      echo -e "• Uses 4-bit quantization for LLM parameters"
      echo -e "• Reduces LLM memory by ~87.5% compared to FP32"
      echo -e "• FP32 computations still used (can be slower than FP16)"
      echo -e "• Estimated memory usage: ${GREEN}~30-40%${NC} of baseline"
      ;;
    max)
      echo -e "${PURPLE}MAX OPTIMIZATION MODE - LOWEST MEMORY USAGE${NC}"
      echo -e "• Combines FP16 and 4-bit quantization"
      echo -e "• Maximum memory savings possible"
      echo -e "• Great for training larger models or increasing batch size"
      echo -e "• Estimated memory usage: ${PURPLE}~20-25%${NC} of baseline"
      ;;
  esac
  
  echo -e "${BLUE}===============================================${NC}"
  echo -e "You'll see detailed memory usage in the logs during training."
  echo -e "Look for lines like: ${CYAN}GPU Memory - LLM forward: X MB, Encoder: Y MB${NC}"
  echo -e "${BLUE}===============================================${NC}"
  echo ""
}

# Setup command based on mode
case $MODE in
  standard)
    echo -e "${RED}=== Running in STANDARD mode (highest memory usage) ===${NC}"
    print_memory_info "standard"
    CMD="$BASE_CMD"
    ;;
  fp16)
    echo -e "${YELLOW}=== Running in FP16 mode (reduced memory usage) ===${NC}"
    print_memory_info "fp16"
    CMD="$BASE_CMD --fp16"
    ;;
  4bit)
    echo -e "${GREEN}=== Running in 4-BIT mode (greatly reduced memory usage) ===${NC}"
    print_memory_info "4bit"
    CMD="$BASE_CMD --use_4bit"
    ;;
  max)
    echo -e "${PURPLE}=== Running in MAX OPTIMIZATION mode (lowest memory usage) ===${NC}"
    print_memory_info "max"
    CMD="$BASE_CMD --fp16 --use_4bit"
    ;;
  *)
    echo "Error: Unknown mode '$MODE'. Use 'standard', 'fp16', '4bit', or 'max'."
    exit 1
    ;;
esac

# Execute the command
echo -e "Running command: ${CYAN}$CMD${NC}"
$CMD 