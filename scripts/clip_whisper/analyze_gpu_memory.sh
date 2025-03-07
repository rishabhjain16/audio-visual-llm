#!/bin/bash
# Script to analyze GPU memory usage of CLIP-Whisper components

# ANSI color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
LLM_PATH="checkpoints/Llama-3.2-1B"
WHISPER_MODEL="openai/whisper-medium"
CLIP_MODEL="openai/clip-vit-base-patch32"
OUTPUT_DIR="outputs/memory_analysis"
COMPARE_4BIT=true
USE_LORA=true
MODALITY="multimodal"  # Default is multimodal (both audio and video)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --modality)
      MODALITY="$2"
      shift 2
      ;;
    --no-4bit-comparison)
      COMPARE_4BIT=false
      shift
      ;;
    --no-lora)
      USE_LORA=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate modality
if [[ "$MODALITY" != "audio" && "$MODALITY" != "video" && "$MODALITY" != "multimodal" ]]; then
  echo -e "${RED}Error: Invalid modality '$MODALITY'. Must be 'audio', 'video', or 'multimodal'.${NC}"
  exit 1
fi

# Print welcome message
echo -e "${BLUE}===============================================${NC}"
echo -e "${CYAN}    GPU MEMORY ANALYZER FOR CLIP-WHISPER      ${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "This tool will analyze GPU memory usage of each component:"

if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "audio" ]]; then
  echo -e "  - ${YELLOW}Whisper${NC} (audio encoder)"
fi
if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "video" ]]; then
  echo -e "  - ${GREEN}CLIP${NC} (video encoder)"
fi
echo -e "  - ${RED}LLM${NC} (language model)"

echo ""
echo -e "Configuration:"
echo -e "  LLM: ${CYAN}$LLM_PATH${NC}"
if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "audio" ]]; then
  echo -e "  Whisper: ${CYAN}$WHISPER_MODEL${NC}"
fi
if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "video" ]]; then
  echo -e "  CLIP: ${CYAN}$CLIP_MODEL${NC}"
fi
echo -e "  Output directory: ${CYAN}$OUTPUT_DIR${NC}"
echo -e "  Compare 4-bit: ${CYAN}$COMPARE_4BIT${NC}"
echo -e "  Use LoRA: ${CYAN}$USE_LORA${NC}"
echo -e "  Modality: ${CYAN}$MODALITY${NC}"
echo -e "${BLUE}===============================================${NC}"

# Build and execute the command
CMD="python scripts/clip_whisper/analyze_memory.py \
  --llm_path $LLM_PATH \
  --output_dir $OUTPUT_DIR \
  --modality $MODALITY"

if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "audio" ]]; then
  CMD="$CMD --whisper_model $WHISPER_MODEL"
fi

if [[ "$MODALITY" == "multimodal" || "$MODALITY" == "video" ]]; then
  CMD="$CMD --clip_model $CLIP_MODEL"
fi

if [ "$COMPARE_4BIT" = true ]; then
  CMD="$CMD --check_4bit"
fi

if [ "$USE_LORA" = true ]; then
  CMD="$CMD --use_lora"
fi

echo -e "${YELLOW}Starting memory analysis... This may take a few minutes.${NC}"
echo -e "Running command: ${CYAN}$CMD${NC}"
echo ""

# Execute the command
$CMD

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Memory analysis completed successfully!${NC}"
  echo -e "Memory usage chart saved to: ${CYAN}$OUTPUT_DIR/memory_usage_chart.png${NC}"
  echo -e "Detailed memory statistics saved to: ${CYAN}$OUTPUT_DIR/memory_stats.json${NC}"
else
  echo -e "${RED}Memory analysis failed with exit code $?${NC}"
  exit 1
fi

# Provide instructions for viewing results
echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${CYAN}             NEXT STEPS                       ${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "To view the memory usage chart:"
echo -e "  1. Open the file: ${CYAN}$OUTPUT_DIR/memory_usage_chart.png${NC}"
echo -e ""
echo -e "To view detailed memory statistics:"
echo -e "  1. Open the file: ${CYAN}$OUTPUT_DIR/memory_stats.json${NC}"
echo -e "  2. Or use the command: ${CYAN}cat $OUTPUT_DIR/memory_stats.json | jq${NC}"
echo -e "${BLUE}===============================================${NC}" 