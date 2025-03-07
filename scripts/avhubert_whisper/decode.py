#!/usr/bin/env python3
"""
Decode script for the AVHuBERT-Whisper model.
"""

import os
import sys
import logging
import argparse
import torch
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.avhubert_whisper.models import AVHuBERTWhisperModel
from src.avhubert_whisper.utils import setup_logging, load_config, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the AVHuBERT-Whisper model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to the audio file for transcription")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to the video file for transcription")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to write transcription (stdout if not specified)")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "both"], default=None,
                        help="Modality to use for inference (defaults to model's default)")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum sequence length for generation")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)
    
    # Set random seed
    set_seed(args.seed)
    
    # Check inputs
    if args.audio_path is None and args.video_path is None:
        logging.error("Either audio_path or video_path must be provided")
        return 1
    
    # Load model
    try:
        logging.info(f"Loading model from {args.model_path}")
        model = AVHuBERTWhisperModel.from_pretrained(args.model_path)
        
        # Set modality if specified
        if args.modality:
            model.modality = args.modality
            logging.info(f"Set modality to: {args.modality}")
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info(f"Model loaded successfully and moved to {device}")
        
        # Load audio data if provided
        audio_tensor = None
        if args.audio_path:
            if not os.path.exists(args.audio_path):
                logging.error(f"Audio file not found: {args.audio_path}")
                return 1
            
            logging.info(f"Loading audio from {args.audio_path}")
            # This loading code depends on your specific audio loading function
            # For simplicity, assuming there's a function to handle this
            # audio_tensor = load_audio(args.audio_path)
            logging.info(f"Audio loaded with shape: {audio_tensor.shape if audio_tensor is not None else None}")
        
        # Load video data if provided
        video_tensor = None
        if args.video_path:
            if not os.path.exists(args.video_path):
                logging.error(f"Video file not found: {args.video_path}")
                return 1
            
            logging.info(f"Loading video from {args.video_path}")
            # This loading code depends on your specific video loading function
            # For simplicity, assuming there's a function to handle this
            # video_tensor = load_video(args.video_path)
            logging.info(f"Video loaded with shape: {video_tensor.shape if video_tensor is not None else None}")
        
        # Move inputs to device
        if audio_tensor is not None:
            audio_tensor = audio_tensor.to(device)
        if video_tensor is not None:
            video_tensor = video_tensor.to(device)
        
        # Generate transcription
        logging.info("Generating transcription...")
        outputs = model.generate(
            audio=audio_tensor,
            video=video_tensor,
            max_length=args.max_length,
            num_beams=args.beam_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        
        # Process generated tokens
        generated_ids = outputs["generated_ids"]
        
        # Decode tokens to text
        tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        if tokenizer:
            transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            transcription = [t.strip() for t in transcription]
        else:
            logging.warning("No tokenizer found in model, unable to decode")
            transcription = ["[Unable to decode tokens without tokenizer]"]
        
        # Output the transcription
        if args.output_file:
            with open(args.output_file, "w") as f:
                for t in transcription:
                    f.write(f"{t}\n")
            logging.info(f"Transcription written to {args.output_file}")
        else:
            print("\nTRANSCRIPTION:")
            for i, t in enumerate(transcription):
                print(f"{i+1}: {t}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 