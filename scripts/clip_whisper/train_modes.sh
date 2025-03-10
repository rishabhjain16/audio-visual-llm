#!/bin/bash
# Helper script to run training with different memory optimization modes

# Default values
DATA_PATH="/home/rishabh/Desktop/Datasets/lrs3/433h_data"
EPOCHS=10
BATCH_SIZE=2
MODALITY="both"
LLM_PATH="checkpoints/Llama-3.2-1B"
MODE="max"  # standard, fp16, 4bit, or max
OUTPUT_DIR="outputs/new_test_clip_whisper"
DEBUG="true"
MAX_SEQ_LEN=1536  # Default is now 1536 to capture more of the 1500 audio frames
CONNECTOR_TYPE="qformer"  # default connector type

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
    --connector_type)
      CONNECTOR_TYPE="$2"
      shift 2
      ;;
    --llm_path)
      LLM_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Base command
BASE_CMD="bash scripts/clip_whisper/train.sh --data_path $DATA_PATH --max_epochs $EPOCHS --batch_size $BATCH_SIZE --modality $MODALITY --output_dir $OUTPUT_DIR --max_seq_len $MAX_SEQ_LEN --connector_type $CONNECTOR_TYPE --llm_path $LLM_PATH"

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

# Print connector information
echo -e "${BLUE}===============================================${NC}"
echo -e "${CYAN}           CONNECTOR TYPE INFORMATION           ${NC}"
echo -e "${BLUE}===============================================${NC}"

case $CONNECTOR_TYPE in
  simple)
    echo -e "${GREEN}SIMPLE CONNECTOR${NC}"
    echo -e "• Basic linear projection from encoder to LLM dimension"
    echo -e "• Minimal memory usage and computational overhead"
    echo -e "• Good baseline approach for stability"
    ;;
  deep)
    echo -e "${YELLOW}DEEP CONNECTOR${NC}"
    echo -e "• Multi-layer projection with residual connections"
    echo -e "• Better representation learning capability"
    echo -e "• Includes LayerNorm for improved training stability"
    echo -e "• Moderate increase in parameters and computation"
    ;;
  conv)
    echo -e "${PURPLE}CONVOLUTIONAL CONNECTOR${NC}"
    echo -e "• Uses 1D convolutions to capture sequence patterns"
    echo -e "• Better handling of local relationships in sequences"
    echo -e "• Good for speech and visual temporal features"
    echo -e "• Moderate increase in computation"
    ;;
  attention)
    echo -e "${CYAN}ATTENTION CONNECTOR${NC}"
    echo -e "• Self-attention based for capturing global relationships"
    echo -e "• Best for long-range dependencies in sequences"
    echo -e "• Includes residual connections and layer norms"
    echo -e "• Higher computation cost but powerful modeling"
    ;;
  adaptive)
    echo -e "${RED}ADAPTIVE CONNECTOR${NC}"
    echo -e "• Dynamically adapts to sequence length using pooling"
    echo -e "• Uses convolution for downsampling longer sequences"
    echo -e "• Includes positional encoding for sequence awareness"
    echo -e "• Best for handling variable-length sequences"
    ;;
  cross_modal)
    echo -e "${BLUE}CROSS-MODAL CONNECTOR${NC}"
    echo -e "• Enables audio and video to attend to each other"
    echo -e "• True multimodal fusion through cross-attention"
    echo -e "• Learns relationships between lip movements and speech"
    echo -e "• Higher memory usage but better multimodal understanding"
    echo -e "• Ideal for audio-visual tasks like lip reading"
    ;;
  qformer)
    echo -e "${PURPLE}QFORMER CONNECTOR${NC}"
    echo -e "• Inspired by BLIP-2 and Flamingo architectures"
    echo -e "• Uses learnable query vectors to extract key information"
    echo -e "• Efficient for long sequences (fixed output length)"
    echo -e "• Strong semantic understanding of multimodal content"
    echo -e "• Recommended for complex audio-visual understanding tasks"
    ;;
  perceiver)
    echo -e "${CYAN}PERCEIVER CONNECTOR${NC}"
    echo -e "• Based on Perceiver-IO architecture from DeepMind"
    echo -e "• Processes inputs through a small set of latent vectors"
    echo -e "• Very memory-efficient for long sequences"
    echo -e "• Can handle extremely long audio-visual sequences"
    echo -e "• Ideal for large-scale multimodal integration"
    ;;
  *)
    echo -e "${RED}UNKNOWN CONNECTOR TYPE: $CONNECTOR_TYPE${NC}"
    echo -e "Defaulting to simple connector"
    ;;
esac

echo -e "${BLUE}===============================================${NC}"
echo ""

$CMD 