#!/usr/bin/env python3
# Copyright (c) 2023-2024 
# All rights reserved.

import os
import sys
import logging
import argparse
from pathlib import Path
import torch

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.trainer.trainer import AVSRTrainer
from src.utils.config import load_config
from src.utils.setup import setup_logging, setup_environment, setup_seed
from src.data.dataset import AVSRDataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSR-LLM model")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the configuration file")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints (overrides config)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--llm_path", type=str, default=None,
                        help="Path to LLM model (overrides config)")
    parser.add_argument("--av_encoder_path", type=str, default=None,
                        help="Path to AV encoder model (overrides config)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more detailed logging and a smaller dataset")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size in config")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N training steps (overrides save_every)")
    parser.add_argument("--log_param_updates", action="store_true",
                        help="Log parameter updates during training")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "both"], default=None,
                        help="Modality to use for training: audio-only, video-only, or both")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup environment
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.data_path:
        config.data.path = args.data_path
    if args.llm_path:
        config.model.llm_path = args.llm_path
    if args.av_encoder_path:
        config.model.av_encoder_path = args.av_encoder_path
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.training.num_epochs = args.max_epochs
    if args.save_every is not None:
        config.training.save_every = args.save_every
    if args.save_steps is not None:
        config.training.save_steps = args.save_steps
    if args.modality is not None:
        config.model.modality = args.modality
        logging.info(f"Setting training modality to: {args.modality}")
    
    # Set parameter update logging
    config.log_param_updates = args.log_param_updates
    
    # Set debug mode
    config.debug = args.debug
    
    # Setup logging
    log_dir = Path(config.training.checkpoint_dir) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_dir / "train.log", level=log_level)
    
    # Log configuration
    logging.info(f"Training with configuration: {config}")
    if args.debug:
        logging.debug("Debug mode enabled - verbose logging and memory tracking enabled")
    
    # Setup environment and seed
    setup_environment()
    setup_seed(args.seed)
    
    # Initialize trainer
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    config.device = device
    
    # Log GPU info before training
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(args.gpu)
        free_mem, total_mem = torch.cuda.mem_get_info(args.gpu)
        free_mem_gb = free_mem / (1024 ** 3)
        total_mem_gb = total_mem / (1024 ** 3)
        used_mem_gb = total_mem_gb - free_mem_gb
        
        logging.info(f"Using GPU: {gpu_props.name} with {total_mem_gb:.2f}GB memory")
        logging.info(f"GPU memory usage before loading model: {used_mem_gb:.2f}GB used / {total_mem_gb:.2f}GB total")
    
    # Initialize trainer with updated interface
    trainer = AVSRTrainer(config, gpu=args.gpu)
    
    # Train the model
    results = trainer.train(debug_mode=args.debug)
    
    # Log final results
    if isinstance(results, dict) and results.get('status') == 'success':
        logging.info("Training completed successfully")
        return 0
    else:
        logging.error("Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 