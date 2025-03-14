#!/usr/bin/env python3
"""
Training script for the ClipWhisperModel.
"""

import os
import sys
import logging

# Configure logging early to suppress specific loggers
for logger_name in ["urllib3", "urllib3.connectionpool", "huggingface_hub", "transformers"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = True

import argparse
import torch
import yaml
from pathlib import Path
import traceback
import random

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
    parser.add_argument("--connector_type", type=str, default="simple",
                        choices=["simple", "deep", "conv", "attention", "adaptive", "cross_modal", "qformer", "perceiver"],
                        help="Type of connector to use for modality projection")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Maximum gradient norm for clipping (lower values like 0.1-0.5 help with larger models)")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_default_logging(output_dir):
    """Set up default logging configuration"""
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Default to INFO level
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    log_file = os.path.join(output_dir, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Silence external libraries
    for logger_name in logging.root.manager.loggerDict:
        if logger_name not in ['root', 'clip_whisper']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info("Default logging configured")


def main():
    args = parse_args()
    
    # Set up default logging
    setup_default_logging(args.output_dir)
    
    config_path = args.config if args.config else None
    config = load_config(config_path) if config_path else {}
    
    # Override config with command-line args
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(config.get("seed", 42))
    random.seed(config.get("seed", 42))
    
    # Set up device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Log effective config
    logging.info(f"Training with config: {config}")
    
    # Use mixed precision if requested
    fp16 = config.get("fp16", False)
    logging.info(f"FP16 training: {fp16}")
    
    # Load model
    model = ClipWhisperModel(
        llm_path=config.get("llm_path", "meta-llama/Llama-2-7b-chat-hf"),
        whisper_model=config.get("whisper_model", "openai/whisper-medium"),
        clip_model=config.get("clip_model", "openai/clip-vit-base-patch32"),
        device=device,
        use_fp16=fp16,
        use_4bit=config.get("use_4bit", False),
        use_lora=not config.get("no_lora", False),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        freeze_encoders=config.get("freeze_encoders", True),
        freeze_llm=config.get("freeze_llm", False),
        modality=config.get("modality", "both"),
        max_seq_len=config.get("max_seq_len", 256),
        fusion_scale=config.get("fusion_scale", 0.5),
        connector_type=config.get("connector_type", "simple"),
    ).to(device)
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    # Estimate number of training samples for scheduler - more robust approach
    try:
        # First try checking for train.tsv file directly in data_path
        train_tsv = os.path.join(config["data_path"], "train.tsv")
        if os.path.exists(train_tsv):
            # Count lines in TSV file (subtract 1 for header if present)
            with open(train_tsv, 'r') as f:
                num_samples = sum(1 for _ in f)
                if num_samples > 0:  # Account for possible header
                    num_samples -= 1
            logging.info(f"Estimated {num_samples} training samples from {train_tsv}")
        else:
            # Try train subdirectory path
            train_dir = os.path.join(config["data_path"], "train")
            train_tsv = os.path.join(train_dir, "train.tsv")
            if os.path.exists(train_tsv):
                with open(train_tsv, 'r') as f:
                    num_samples = sum(1 for _ in f)
                    if num_samples > 0:  # Account for possible header
                        num_samples -= 1
                logging.info(f"Estimated {num_samples} training samples from {train_tsv}")
            else:
                # Try data subdirectory path
                data_dir = os.path.join(config["data_path"], "data")
                train_tsv = os.path.join(data_dir, "train.tsv")
                if os.path.exists(train_tsv):
                    with open(train_tsv, 'r') as f:
                        num_samples = sum(1 for _ in f)
                        if num_samples > 0:  # Account for possible header
                            num_samples -= 1
                    logging.info(f"Estimated {num_samples} training samples from {train_tsv}")
                else:
                    # Use directory size or default
                    num_samples = len(os.listdir(config["data_path"])) if os.path.isdir(config["data_path"]) else 1000
                    logging.info(f"Could not find train.tsv file. Using estimated {num_samples} samples.")
    except Exception as e:
        num_samples = 1000  # Fallback to a reasonable default
        logging.warning(f"Could not determine dataset size: {str(e)}. Using default of {num_samples} samples.")
    
    steps_per_epoch = num_samples // config.get("batch_size", 4)
    total_steps = config.get("max_epochs", 10) * steps_per_epoch
    
    logging.info(f"Scheduler setup: {steps_per_epoch} steps per epoch, {total_steps} total steps")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Add gradient clipping to stabilize training
    max_grad_norm = config.get("max_grad_norm", 1.0)
    logging.info(f"Using gradient clipping with grad_clip={max_grad_norm}")
    
    # Create dataloaders - with robust error handling
    try:
        logging.info(f"Creating dataloaders from {config['data_path']}")
        train_dataloader, val_dataloader = create_simple_dataloaders(
            data_path=config["data_path"],
            batch_size=config.get("batch_size", 4),
            num_workers=config.get("num_workers", 4),
            config=config
        )
        logging.info(f"Successfully created dataloaders")
    except Exception as e:
        logging.error(f"Error creating dataloaders: {e}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Failed to create dataloaders from {config['data_path']}: {e}")
    
    # Initialize trainer
    trainer = ClipWhisperTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        max_epochs=config.get("max_epochs", 10),
        output_dir=config["output_dir"],
        device=device,
        fp16=fp16,
        grad_accum_steps=config.get("grad_accumulation_steps", 1),
        log_interval=config.get("log_every", 10),
        save_every=config.get("save_every", 0),
        save_steps=config.get("save_steps", None),
        grad_clip=max_grad_norm,
        warmup_steps=config.get("warmup_steps", 0),
        log_param_updates=config.get("log_param_updates", False),
    )
    
    # Start training
    if config.get("resume_from"):
        logging.info(f"Resuming training from {config['resume_from']}")
    trainer.train()
    
    logging.info("Training completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 