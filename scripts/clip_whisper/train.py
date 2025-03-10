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
    parser.add_argument("--llm_path", type=str, default=None,
                        help="Path or name of the LLM model")
    parser.add_argument("--whisper_model", type=str, default=None,
                        help="Name or path of Whisper model")
    parser.add_argument("--clip_model", type=str, default=None,
                        help="Name or path of CLIP model")
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
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--use_4bit", action="store_true", default=None,
                        help="Use 4-bit quantization for the LLM (requires bitsandbytes)")
    parser.add_argument("--no_lora", action="store_true", 
                        help="Disable LoRA fine-tuning")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_param_updates", action="store_true",
                        help="Log parameter updates during training")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Maximum sequence length for encoder output (overrides config)")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    parser.add_argument("--connector_type", type=str, default="simple",
                        choices=["simple", "deep", "conv", "attention", "adaptive", "cross_modal", "qformer", "perceiver"],
                        help="Type of connector to use for modality projection")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=numeric_level,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")
    
    # Setup logging
    setup_logging()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    logging.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.output_dir:
        config["training"]["checkpoint_dir"] = args.output_dir
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.llm_path:
        config["model"]["llm_path"] = args.llm_path
    if args.whisper_model:
        config["model"]["whisper_model"] = args.whisper_model
    if args.clip_model:
        config["model"]["clip_model"] = args.clip_model
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.max_epochs:
        config["training"]["num_epochs"] = args.max_epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.modality:
        config["model"]["modality"] = args.modality
    if args.fp16 is not None:
        config["model"]["use_fp16"] = args.fp16
    if args.use_4bit is not None:
        config["model"]["use_4bit"] = args.use_4bit
    if args.no_lora:
        config["model"]["use_lora"] = False
    if args.log_interval:
        config["training"]["log_interval"] = args.log_interval
    if args.save_every:
        config["training"]["save_every"] = args.save_every
    if args.save_steps:
        config["training"]["save_steps"] = args.save_steps
    if args.max_seq_len is not None:
        config["model"]["max_seq_len"] = args.max_seq_len
        logging.info(f"Overriding max_seq_len with {args.max_seq_len}")
    
    if args.connector_type:
        config["model"]["connector_type"] = args.connector_type
        logging.info(f"Using connector type: {args.connector_type}")
    
    # Log the effective max_seq_len
    logging.info(f"Using max_seq_len: {config['model']['max_seq_len']}")
    
    # Create output directory
    output_dir = config["training"]["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Log basic info
    logging.info("STARTING CLIP-WHISPER TRAINING")
    logging.info("=" * 80)
    logging.info(f"Data path: {config['data']['path']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"Selected modality: {config['model']['modality']}")
    logging.info(f"Batch size: {config['data']['batch_size']}")
    logging.info(f"Max epochs: {config['training']['num_epochs']}")
    logging.info(f"FP16: {config['model'].get('use_fp16', False)}")
    logging.info(f"4-bit quantization: {config['model'].get('use_4bit', False)}")
    logging.info(f"LoRA: {config['model'].get('use_lora', True)}")
    logging.info(f"Log parameter updates: {args.log_param_updates}")
    
    try:
        # Create dataloaders
        train_dataloader, val_dataloader = create_simple_dataloaders(
            data_path=args.data_path,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"].get("num_workers", 2),
            config=config
        )
        
        # Log dataloader information
        logging.info(f"Created dataloaders: train={len(train_dataloader)}, val={len(val_dataloader) if val_dataloader else 0}")
        
        # Create model with explicit logging of dimensions
        logging.info("Creating ClipWhisperModel...")
        model = ClipWhisperModel(
            llm_path=config["model"]["llm_path"],
            whisper_model=config["model"]["whisper_model"],
            clip_model=config["model"]["clip_model"],
            device=device,
            use_fp16=config["model"].get("use_fp16", False),
            use_4bit=config["model"].get("use_4bit", False),
            use_lora=config["model"].get("use_lora", True),
            lora_r=config["model"].get("lora_r", 16),
            lora_alpha=config["model"].get("lora_alpha", 32),
            lora_dropout=config["model"].get("lora_dropout", 0.05),
            freeze_encoders=config["model"].get("freeze_encoders", True),
            freeze_llm=config["model"].get("freeze_llm", False),
            modality=config["model"]["modality"],
            max_seq_len=config["model"].get("max_seq_len", 256),
            fusion_scale=config["model"].get("fusion_scale", 0.5),
        )
        
        # Create trainer
        trainer = ClipWhisperTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
            max_epochs=config["training"]["num_epochs"],
            output_dir=output_dir,
            device=device,
            fp16=config["model"].get("use_fp16", False),
            grad_accum_steps=config["training"].get("grad_accum_steps", 4),
            log_interval=config["training"].get("log_interval", 10),
            save_every=config["training"].get("save_every", 1),
            save_steps=config["training"].get("save_steps", None),
            max_grad_norm=config["training"].get("max_grad_norm", 0.5),
            warmup_steps=config["training"].get("warmup_steps", 0),
            log_param_updates=args.log_param_updates,
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logging.info(f"Resuming training from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        
        # Train the model
        trainer.train()
        
        logging.info("Training completed successfully")
        return 0
    
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 