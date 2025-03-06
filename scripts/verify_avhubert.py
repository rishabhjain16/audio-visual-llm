#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.av_hubert import AVHuBERTEncoder
from src.utils.setup import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Verify AV-HuBERT model loading and basic operation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AV-HuBERT checkpoint")
    parser.add_argument("--layer", type=int, default=-1, help="Which transformer layer to extract features from")
    parser.add_argument("--use_audio", action="store_true", help="Test audio modality")
    parser.add_argument("--use_video", action="store_true", help="Test video modality")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Default to both modalities if none specified
    if not args.use_audio and not args.use_video:
        args.use_audio = True
        args.use_video = True
    
    logging.info(f"Checkpoint path: {args.checkpoint}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Using audio: {args.use_audio}")
    logging.info(f"Using video: {args.use_video}")
    
    # Verify the checkpoint file exists
    if not os.path.exists(args.checkpoint):
        logging.error(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        # Try to list files in the directory
        parent_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(parent_dir):
            files = os.listdir(parent_dir)
            logging.error(f"Files in directory: {files[:10]}")
            if len(files) > 10:
                logging.error(f"... and {len(files) - 10} more files")
        else:
            logging.error(f"Parent directory does not exist: {parent_dir}")
        return 1
    
    logging.info("Checkpoint file exists!")
    
    try:
        # Try to load the model
        logging.info("Loading AV-HuBERT encoder...")
        model = AVHuBERTEncoder(
            checkpoint_path=args.checkpoint,
            layer=args.layer,
            use_audio=args.use_audio,
            use_video=args.use_video,
            freeze=True
        )
        
        # Move to device
        model = model.to(args.device)
        
        logging.info(f"Model loaded successfully! Embedding dim: {model.embedding_dim}")
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        
        # Create dummy audio input if needed
        audio_input = None
        if args.use_audio:
            audio_input = torch.randn(batch_size, seq_len, 80, device=args.device)
            logging.info(f"Created dummy audio input with shape: {audio_input.shape}")
        
        # Create dummy video input if needed
        video_input = None
        if args.use_video:
            # Create grayscale frames [B, T, C, H, W]
            video_input = torch.randn(batch_size, seq_len, 1, 96, 96, device=args.device)
            logging.info(f"Created dummy video input with shape: {video_input.shape}")
        
        # Run forward pass
        logging.info("Running forward pass...")
        with torch.no_grad():
            output = model(audio=audio_input, video=video_input)
        
        if output is None:
            logging.error("Forward pass returned None!")
            return 1
        
        # Check if output is a dictionary (which happens with some AV-HuBERT implementations)
        if isinstance(output, dict):
            logging.info(f"Forward pass returned a dictionary with keys: {list(output.keys())}")
            # Try to find encoder output in the dictionary
            if "encoder_out" in output:
                encoder_out = output["encoder_out"]
                if isinstance(encoder_out, list):
                    encoder_out = encoder_out[0]  # Usually the first element is what we want
                logging.info(f"Found encoder_out with shape: {encoder_out.shape}")
            else:
                # Just log the first tensor we find
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        logging.info(f"Found tensor '{key}' with shape: {value.shape}")
                        break
        else:
            # Output is a tensor
            logging.info(f"Forward pass successful! Output shape: {output.shape}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error verifying AV-HuBERT model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 