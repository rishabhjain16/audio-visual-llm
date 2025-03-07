#!/usr/bin/env python3
import torch
import os
import logging
import argparse
from pathlib import Path
from src.models.avsr_llm import AVSRLLM
from src.utils.config import load_config, dict_to_object
from src.utils.media import load_audio as load_audio_util
from src.utils.media import load_video as load_video_util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AVSR-LLM Demo")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--audio_path", type=str, help="Path to audio file")
    parser.add_argument("--video_path", type=str, help="Path to video file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Ensure the config has model and training attributes
    if not hasattr(config, 'model'):
        config.model = dict_to_object({})
    if not hasattr(config, 'training'):
        config.training = dict_to_object({})
    
    # Force settings for inference
    config.model.fusion_dim = 2048  # Match LLM dimension
    config.model.llm_dim = 2048     # Llama-3.2-1B dimension
    config.model.use_lora = True    # Use LoRA for efficiency
    
    # Create model
    logger.info("Creating model...")
    try:
        model = AVSRLLM(config=config, device=device)
        model.eval()  # Set to evaluation mode
        logger.info(f"Model created with fusion_dim={model.fusion_dim}, llm_dim={model.llm_dim}")
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Create dummy inputs 
    logger.info("Creating dummy inputs...")
    # Create dummy 1-second audio signal
    audio = torch.randn(1, 16000).to(device) if args.audio_path is None else load_audio(args.audio_path, device)
    
    # Create dummy 1-second video (25 frames of 96x96)
    video = torch.randn(1, 25, 1, 96, 96).to(device) if args.video_path is None else load_video(args.video_path, device)
    
    # Generate text
    try:
        with torch.no_grad():
            logger.info("Generating text...")
            outputs = model.generate(audio=audio, video=video)
            
            # Handle different output formats
            if isinstance(outputs, list):
                output_text = outputs[0] if outputs else "No output generated"
            else:
                output_text = str(outputs)
                
            logger.info(f"Generated text: {output_text}")
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
def load_audio(audio_path, device):
    """Load audio file using the proper utility"""
    try:
        return load_audio_util(audio_path, device=device)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        logger.info("Using dummy audio instead")
        return torch.randn(1, 16000).to(device)

def load_video(video_path, device):
    """Load video file using the proper utility"""
    try:
        return load_video_util(video_path, device=device)
    except Exception as e:
        logger.error(f"Error loading video: {e}")
        logger.info("Using dummy video instead")
        return torch.randn(1, 25, 1, 96, 96).to(device)

if __name__ == "__main__":
    main() 