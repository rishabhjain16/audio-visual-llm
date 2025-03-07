#!/usr/bin/env python3
"""
Training script for the AVHuBERT-Whisper model.
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

from src.avhubert_whisper.models import AVHuBERTWhisperModel
from src.avhubert_whisper.trainer import AVHuBERTWhisperTrainer
from src.avhubert_whisper.utils import setup_logging, load_config, set_seed, AVHuBERTWhisperConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train AVHuBERT-Whisper model")
    parser.add_argument("--config", type=str, default="configs/avhubert_whisper.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/avhubert_whisper",
                        help="Directory to save outputs")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--llm_path", type=str, default=None,
                        help="Path or name of the LLM model")
    parser.add_argument("--whisper_model", type=str, default=None,
                        help="Name or path of Whisper model")
    parser.add_argument("--avhubert_path", type=str, default=None,
                        help="Path to the AVHuBERT checkpoint")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--modality", type=str, default=None, 
                        choices=["audio", "video", "both"],
                        help="Which modalities to use")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_param_updates", action="store_true",
                        help="Log parameter updates during training")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.data_path:
        config["data_path"] = args.data_path
    if args.llm_path:
        config["llm_path"] = args.llm_path
    if args.whisper_model:
        config["whisper_model"] = args.whisper_model
    if args.avhubert_path:
        config["avhubert_path"] = args.avhubert_path
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.modality:
        config["modality"] = args.modality
    if args.fp16 is not None:
        config["use_fp16"] = args.fp16
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)
    
    # Log configuration
    logging.info("STARTING AVHUBERT-WHISPER TRAINING")
    logging.info("=" * 80)
    logging.info(f"Data path: {config['data_path']}")
    logging.info(f"Output directory: {config['output_dir']}")
    logging.info(f"Device: {device}")
    logging.info(f"Selected modality: {config['modality']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Max epochs: {config['max_epochs']}")
    logging.info(f"FP16: {config['use_fp16']}")
    logging.info(f"Log parameter updates: {args.log_param_updates}")
    
    try:
        # Create dataset and dataloaders
        # This part depends on your specific dataset implementation
        # For simplicity, assuming you already have them implemented
        train_dataloader = None  # Replace with your actual dataloader
        val_dataloader = None    # Replace with your actual dataloader
        
        # Create model
        logging.info("Creating AVHuBERT-Whisper model...")
        model = AVHuBERTWhisperModel(
            llm_path=config["llm_path"],
            whisper_model=config["whisper_model"],
            avhubert_path=config["avhubert_path"],
            device=device,
            use_fp16=config["use_fp16"],
            use_lora=config.get("use_lora", True),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            freeze_encoders=config.get("freeze_encoders", True),
            freeze_llm=config.get("freeze_llm", False),
            modality=config["modality"],
            max_seq_len=config.get("max_seq_len", 256),
            fusion_scale=config.get("fusion_scale", 0.5),
        )
        
        # Create trainer
        logging.info("Creating trainer...")
        trainer = AVHuBERTWhisperTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
            max_epochs=config["max_epochs"],
            output_dir=config["output_dir"],
            device=device,
            fp16=config["use_fp16"],
            grad_accum_steps=config.get("grad_accum_steps", 4),
            log_interval=config.get("log_interval", 10),
            save_every=args.save_every or config.get("save_every", 1),
            save_steps=args.save_steps or config.get("save_steps", None),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            warmup_steps=config.get("warmup_steps", 0),
            log_param_updates=args.log_param_updates,
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume_from:
            start_epoch = trainer.resume_from_checkpoint(args.resume_from)
        
        # Train model
        logging.info(f"Starting training from epoch {start_epoch}...")
        trainer.train()
        
        logging.info("Training completed successfully!")
        return 0
    
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 