#!/usr/bin/env python3
"""
Training script for the ClipWhisperModel.
"""

import os
import sys
import logging
import argparse
import torch
import yaml
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.clip_whisper.models import ClipWhisperModel
from src.clip_whisper.trainer import ClipWhisperTrainer
from src.clip_whisper.data import create_simple_dataloaders
from src.utils.setup import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train ClipWhisperModel")
    parser.add_argument("--config", type=str, default="configs/clip_whisper.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/clip_whisper",
                        help="Directory to save outputs")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Path or name of the LLM model")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-medium",
                        help="Path or name of the Whisper model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="Path or name of the CLIP model")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training (lower values like 2-4 improve stability)")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "both"], default="both",
                        help="Modality to use for training: audio-only, video-only, or both")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N training steps (overrides save_every)")
    parser.add_argument("--log_param_updates", action="store_true",
                        help="Log parameter updates during training")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more logging")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Set up GPU device
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(os.path.join(output_dir, "train.log"), level=log_level)
    
    # Log startup information
    logging.info("=" * 80)
    logging.info("STARTING CLIP-WHISPER TRAINING")
    logging.info("=" * 80)
    logging.info(f"Data path: {args.data_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"Selected modality: {args.modality}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Max epochs: {args.max_epochs}")
    logging.info(f"FP16: {args.fp16}")
    logging.info(f"Log parameter updates: {args.log_param_updates}")
    
    try:
        # Create dataloaders
        train_dataloader, val_dataloader = create_simple_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=config["data"].get("num_workers", 2),
            config=config
        )
        
        # Log dataloader information
        logging.info(f"Created dataloaders: train={len(train_dataloader)}, val={len(val_dataloader) if val_dataloader else 0}")
        
        # Create model with explicit logging of dimensions
        logging.info("Creating ClipWhisperModel...")
        model = ClipWhisperModel(
            llm_path=args.llm_path,
            whisper_model=args.whisper_model,
            clip_model=args.clip_model,
            device=device,
            use_fp16=args.fp16,
            modality=args.modality,
            max_seq_len=config["data"].get("max_seq_len", 256),
        )
        
        # Create trainer
        logging.info("Creating ClipWhisperTrainer...")
        trainer = ClipWhisperTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            output_dir=output_dir,
            device=device,
            fp16=args.fp16,
            save_interval=args.save_every,
        )
        
        # Train model
        logging.info("Starting training...")
        results = trainer.train()
        
        # Log final results
        logging.info("=" * 80)
        logging.info("TRAINING COMPLETED")
        logging.info(f"Final train loss: {results['train_losses'][-1]:.6f}")
        if results['val_losses']:
            logging.info(f"Final validation loss: {results['val_losses'][-1]:.6f}")
        logging.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logging.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 